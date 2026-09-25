"""
ML Challenge 2026 - Business Entity Resolution End-to-End Pipeline
Handles:
1. Normalization (US, India, France)
2. Blocking / Candidate Generation
3. Feature Engineering (RapidFuzz name/address similarities)
4. Matching Model (Calibrated LightGBM with macro F_0.5 threshold sweep)
5. Output Generation (matching_results.tsv and candidate_pairs.tsv)
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

# Pipeline modules
from normalize import normalize_business_name, normalize_address, get_phonetic_keys
from metrics import evaluate_macro_f05, evaluate_blocking_recall
from features import compute_pairwise_features
from train_gbm import train_matching_gbm, sweep_macro_f05_threshold
from blocking import CandidateGenerator, format_candidate_pairs_file


def load_and_preprocess_s1(filepath: str, max_records: int = None) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Loads Source 1 TSV and pre-computes normalized fields and tokens.
    """
    all_s1_ids = []
    s1_records = {}

    with open(filepath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""

            clean_name, stripped_name, name_tokens = normalize_business_name(name)
            clean_addr, landmark, addr_tokens, num_tokens = normalize_address(addr, country)

            all_s1_ids.append(eid)
            s1_records[eid] = {
                "name": name,
                "clean_name": clean_name,
                "stripped_name": stripped_name,
                "name_tokens": name_tokens,
                "clean_addr": clean_addr,
                "landmark": landmark,
                "has_landmark": 1 if landmark else 0,
                "addr_tokens": addr_tokens,
                "num_tokens": num_tokens,
                "country": country
            }
            if max_records and len(all_s1_ids) >= max_records:
                break

    return all_s1_ids, s1_records


def generate_candidates(
    s1_records: Dict[str, Dict],
    s2_path: str,
    s3_path: str,
    top_k: int = 50,
    min_score: float = 2.0
) -> Tuple[Dict[str, Set[str]], Dict[str, List[Tuple[float, str]]], CandidateGenerator]:
    """
    Runs multi-channel inverted-index blocking across Source 2 and Source 3.
    """
    generator = CandidateGenerator(top_k=top_k, min_score=min_score)
    generator.index_reference_entities(s1_records)

    # Dictionary of min-heaps for each S1 entity: s1_id -> list of (score, cand_id)
    from collections import defaultdict
    candidates_heap = defaultdict(list)

    print(f"Scanning {os.path.basename(s2_path)} + {os.path.basename(s3_path)} for candidate generation "
          f"(single worker pool, indices sent once)...", flush=True)
    t0 = time.time()
    generator.scan_candidate_pools([s2_path, s3_path], candidates_heap)
    print(f"Finished both source scans in {time.time() - t0:.1f}s.", flush=True)

    # Convert heaps to candidate sets
    candidates = {}
    for s1_id in s1_records.keys():
        heap = candidates_heap.get(s1_id, [])
        candidates[s1_id] = {cid for _, cid in heap}

    return candidates, candidates_heap, generator


def build_pairwise_features(
    s1_records: Dict[str, Dict],
    candidates: Dict[str, Set[str]],
    s2_path: str,
    s3_path: str,
    ground_truth: Dict[str, Set[str]] = None
) -> pd.DataFrame:
    """
    Builds the pairwise feature matrix between S1 entities and candidate records.
    """
    # Collect all needed candidate IDs
    needed_cand_ids = set()
    for cands in candidates.values():
        needed_cand_ids.update(cands)

    print(f"Loading metadata for {len(needed_cand_ids):,} unique candidate records...", flush=True)
    cand_records = {}

    for src_path in [s2_path, s3_path]:
        with open(src_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                cid = parts[0]
                if cid in needed_cand_ids:
                    name = parts[1] if len(parts) > 1 else ""
                    addr = parts[2] if len(parts) > 2 else ""
                    country = parts[3] if len(parts) > 3 else ""

                    clean_name, stripped_name, name_tokens = normalize_business_name(name)
                    clean_addr, landmark, addr_tokens, num_tokens = normalize_address(addr, country)

                    cand_records[cid] = {
                        "clean_name": clean_name,
                        "stripped_name": stripped_name,
                        "name_tokens": name_tokens,
                        "clean_addr": clean_addr,
                        "landmark": landmark,
                        "has_landmark": 1 if landmark else 0,
                        "addr_tokens": addr_tokens,
                        "num_tokens": num_tokens,
                        "country": country
                    }
                    if len(cand_records) >= len(needed_cand_ids):
                        break

    print("Computing pairwise RapidFuzz features...", flush=True)
    rows = []
    for s1_id, cand_set in candidates.items():
        s1 = s1_records[s1_id]
        true_set = ground_truth.get(s1_id, set()) if ground_truth else None

        for cid in cand_set:
            if cid not in cand_records:
                continue
            cand = cand_records[cid]

            feats = compute_pairwise_features(
                s1_name_clean=s1["clean_name"],
                s1_name_stripped=s1["stripped_name"],
                s1_name_tokens=s1["name_tokens"],
                s1_addr_clean=s1["clean_addr"],
                s1_addr_tokens=s1["addr_tokens"],
                s1_num_tokens=s1["num_tokens"],
                s1_country=s1["country"],
                s1_has_landmark=s1["has_landmark"],
                cand_id=cid,
                cand_name_clean=cand["clean_name"],
                cand_name_stripped=cand["stripped_name"],
                cand_name_tokens=cand["name_tokens"],
                cand_addr_clean=cand["clean_addr"],
                cand_addr_tokens=cand["addr_tokens"],
                cand_num_tokens=cand["num_tokens"],
                cand_country=cand["country"],
                cand_has_landmark=cand["has_landmark"]
            )
            feats["source1_entity_id"] = s1_id
            feats["candidate_id"] = cid

            if true_set is not None:
                feats["label"] = 1 if cid in true_set else 0

            rows.append(feats)

    df_features = pd.DataFrame(rows)
    print(f"Constructed feature dataframe with {len(df_features):,} pairs and {len(df_features.columns)} columns.", flush=True)
    return df_features


def run_validation_pipeline(data_dir: str, top_k: int = 50, num_s1: int = 2000):
    """
    Executes Phase 1, Phase 2, and Phase 3 on a held-out validation slice.
    """
    train_dir = os.path.join(data_dir, "train")
    print(f"\n--- 1. Loading {num_s1} Validation S1 Entities & Ground Truth ---", flush=True)

    # 1. Load Ground Truth for num_s1 entities
    gt_map = {}
    s1_needed = set()
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            s1_id = p[0]
            matches = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            gt_map[s1_id] = matches
            s1_needed.add(s1_id)
            if len(gt_map) >= num_s1:
                break

    # 2. Load S1 records for EXACTLY the same s1_needed entities
    all_s1_ids = []
    s1_records = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in s1_needed:
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""

                clean_name, stripped_name, name_tokens = normalize_business_name(name)
                clean_addr, landmark, addr_tokens, num_tokens = normalize_address(addr, country)

                all_s1_ids.append(eid)
                s1_records[eid] = {
                    "name": name,
                    "clean_name": clean_name,
                    "stripped_name": stripped_name,
                    "name_tokens": name_tokens,
                    "clean_addr": clean_addr,
                    "landmark": landmark,
                    "has_landmark": 1 if landmark else 0,
                    "addr_tokens": addr_tokens,
                    "num_tokens": num_tokens,
                    "country": country
                }
                if len(s1_records) >= len(s1_needed):
                    break

    # Phase 1: Candidate Generation
    print("\n--- 2. Phase 1: Candidate Generation (Blocking) ---", flush=True)
    candidates_all, candidates_heap, generator = generate_candidates(
        s1_records=s1_records,
        s2_path=os.path.join(train_dir, "train_source2.tsv"),
        s3_path=os.path.join(train_dir, "train_source3.tsv"),
        top_k=top_k,
        min_score=2.0
    )

    # A4 Sweep: min_score in [3.0, 2.5, 2.0]
    print("\n=== MIN_SCORE SWEEP [2.0, 2.5, 3.0] ON VAL SPLIT (A4) ===", flush=True)
    sweep_evals = {}
    for ms in [3.0, 2.5, 2.0]:
        cands_at_ms = {
            s1_id: {cid for sc, cid in candidates_heap.get(s1_id, []) if sc >= ms}
            for s1_id in s1_records.keys()
        }
        b_eval = evaluate_blocking_recall(gt_map, cands_at_ms)
        sweep_evals[ms] = (b_eval, cands_at_ms)
        print(f"min_score={ms:.1f} -> Recall Ceiling: {b_eval['recall_ceiling']*100:.2f}%, "
              f"Avg Candidates/Entity: {b_eval['avg_candidates_per_entity']:.2f}, "
              f"Captured: {b_eval['captured_matches']:,}/{b_eval['total_true_matches']:,}", flush=True)

    # Use canonical min_score = 3.0 for primary reporting and downstream training
    blocking_eval, candidates = sweep_evals[3.0]
    print("\n=== BLOCKING RECALL CEILING (canonical min_score=3.0) ===")
    for k, v in blocking_eval.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:,}")

    # A5: Missed match analysis
    missed_pairs = []
    for s1_id, true_matches in gt_map.items():
        cands = candidates.get(s1_id, set())
        missed = true_matches - cands
        for m in missed:
            missed_pairs.append((s1_id, m))

    print(f"\n=== MISSED MATCH ANALYSIS (A5) === (Total missed at min_score=3.0: {len(missed_pairs):,})")
    if missed_pairs:
        print("Sample of 20 missed true match pairs for failure mode categorization:")
        for idx, (s1_id, cand_id) in enumerate(missed_pairs[:20], 1):
            s1_info = s1_records.get(s1_id, {})
            print(f"  [{idx}] S1={s1_id} ('{s1_info.get('name', '')}', '{s1_info.get('clean_addr', '')}') vs Cand={cand_id}")

    # Phase 2: Feature Engineering
    print("\n--- 3. Phase 2: Pairwise Feature Construction ---", flush=True)
    df_features = build_pairwise_features(
        s1_records=s1_records,
        candidates=candidates,
        s2_path=os.path.join(train_dir, "train_source2.tsv"),
        s3_path=os.path.join(train_dir, "train_source3.tsv"),
        ground_truth=gt_map
    )

    # Phase 3: LightGBM Training & Threshold Sweeping
    print("\n--- 4. Phase 3: LightGBM Training & Threshold Optimization ---", flush=True)
    from train_val_eval import evaluate_and_train_gbm
    
    model, best_thresh, best_f05, feature_cols = evaluate_and_train_gbm(
        val_candidates=candidates,
        ground_truth=gt_map,
        df_features=df_features
    )

    # Save artifacts
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    import joblib, json
    joblib.dump(model, os.path.join(artifacts_dir, "lgbm_model.pkl"))
    with open(os.path.join(artifacts_dir, "config.json"), "w") as f:
        json.dump({"threshold": best_thresh, "feature_cols": feature_cols, "val_f05": best_f05}, f, indent=2)
    print(f"\nSaved trained model and config to {artifacts_dir}")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution End-to-End Pipeline")
    parser.add_argument("--mode", choices=["validate", "test"], default="validate", help="Pipeline execution mode")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_data = os.path.join(repo_root, "dataset", "student_resource", "dataset")
    default_out = os.path.join(repo_root, "output")
    parser.add_argument("--data-dir", default=default_data)
    parser.add_argument("--output-dir", default=default_out)
    parser.add_argument("--top-k", type=int, default=50, help="Candidate pool size per entity")
    parser.add_argument("--num-val-s1", type=int, default=2000, help="Number of S1 validation entities")
    args = parser.parse_args()

    print(f"=== Starting Pipeline in {args.mode.upper()} Mode ===", flush=True)
    if args.mode == "validate":
        run_validation_pipeline(data_dir=args.data_dir, top_k=args.top_k, num_s1=args.num_val_s1)
    else:
        print("Test mode selected.")


if __name__ == "__main__":
    main()
