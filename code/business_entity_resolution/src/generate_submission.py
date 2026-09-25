"""
Generate Leaderboard Submission:
1. Blocking / Candidate Generation over test set -> output/candidate_pairs.tsv
2. Feature extraction on candidate pairs
3. Model inference + optimal thresholding -> output/matching_results.tsv
4. Automated verification with utils/validate_submission.py
"""

import os
import sys
import time
import json
import joblib
import heapq
import numpy as np
import pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_business_name, normalize_address, get_phonetic_keys
from features import compute_pairwise_features
from blocking import format_candidate_pairs_file, _scan_chunk_worker, _merge_heaps

COMMON_TOKENS = {
    "the", "and", "of", "in", "at", "for", "on", "a", "an", "to", "by", "with", "from",
    "co", "inc", "llc", "ltd", "corp", "corporation", "pvt", "limited", "private", "services",
    "enterprises", "company", "group", "sarl", "sas", "sa", "associates", "trading", "industries",
    "hotel", "restaurant", "store", "shop", "center", "centre", "solutions", "international"
}


def run_test_submission(
    model_path: str,
    config_path: str,
    data_dir: str,
    output_dir: str,
    top_k: int = 50,
    min_score: float = 3.0
):
    print("="*60)
    print("=== GENERATING LEADERBOARD SUBMISSION ===")
    print("="*60, flush=True)

    # 1. Load trained model and configuration
    print(f"Loading model from {model_path}...", flush=True)
    model = joblib.load(model_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    threshold = config["threshold"]
    feature_cols = config["feature_cols"]
    print(f"Loaded model successfully. Decision threshold = {threshold:.2f}", flush=True)

    test_dir = os.path.join(data_dir, "test")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load and index all test_source1 entities
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"\n1. Loading and indexing {s1_path}...", flush=True)
    t0 = time.time()

    all_s1_ids = []
    s1_records = {}

    token_index = defaultdict(list)
    compact_name_index = defaultdict(list)
    phonetic_index = defaultdict(list)
    prefix_index = defaultdict(list)
    st_num_addr_index = defaultdict(list)
    postal_index = defaultdict(list)

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            eid = p[0]
            name = p[1] if len(p) > 1 else ""
            addr = p[2] if len(p) > 2 else ""
            country = p[3] if len(p) > 3 else ""

            all_s1_ids.append(eid)
            cn, sn, nt = normalize_business_name(name)
            ca, lm, at, nums = normalize_address(addr, country)
            pks = get_phonetic_keys(nt[:1])
            compact = "".join(nt)
            p4 = sn[:4] if len(sn) >= 4 else sn

            pc = {n for n in nums if len(n) in (5, 6)}
            st_nums = nums - pc

            s1_records[eid] = {
                "clean_name": cn, "stripped_name": sn, "name_tokens": nt,
                "clean_addr": ca, "has_landmark": 1 if lm else 0,
                "addr_tokens": at, "num_tokens": nums, "country": country
            }

            # Indices
            for t in set(nt):
                if t not in COMMON_TOKENS and len(t) >= 3:
                    token_index[t].append(eid)

            if len(compact) >= 4:
                compact_name_index[compact].append(eid)

            for pk in pks:
                phonetic_index[pk].append(eid)

            if p4 and len(p4) >= 3:
                prefix_index[p4].append(eid)

            # Postal code index
            for pc_val in pc:
                postal_index[pc_val].append(eid)

            if st_nums:
                for sn_val in st_nums:
                    for at_val in at:
                        if len(at_val) >= 3 and not at_val.isdigit():
                            st_num_addr_index[(sn_val, at_val)].append(eid)

    print(f"Indexed {len(all_s1_ids):,} Test S1 entities in {time.time() - t0:.1f}s.", flush=True)

    # Prune ultra high-frequency tokens (appearing in > 300 entities) to prevent scanning bottlenecks
    orig_tok = len(token_index)
    token_index = {t: eids for t, eids in token_index.items() if len(eids) <= 300}
    prefix_index = {p: eids for p, eids in prefix_index.items() if len(eids) <= 300}
    st_num_addr_index = {k: eids for k, eids in st_num_addr_index.items() if len(eids) <= 200}
    print(f"Pruned ultra high-frequency tokens from {orig_tok:,} to {len(token_index):,} distinctive tokens (thresholds: 300/300/200).", flush=True)

    # 3. Stream test_source2 and test_source3 to generate candidates
    candidates_heap = defaultdict(list)

    def scan_test_pool(filepath):
        fname = os.path.basename(filepath)
        t_start = time.time()
        file_size = os.path.getsize(filepath)
        num_workers = max(1, min((os.cpu_count() or 4) - 2, 14))
        print(f"\nScanning {fname} with {num_workers} parallel CPU workers...", flush=True)

        chunk_size = file_size // num_workers
        chunk_args = []
        indices = (
            dict(token_index),
            dict(compact_name_index),
            dict(phonetic_index),
            dict(prefix_index),
            dict(st_num_addr_index),
            dict(postal_index)
        )
        for i in range(num_workers):
            start = i * chunk_size
            end = file_size if i == num_workers - 1 else (i + 1) * chunk_size
            chunk_args.append((filepath, start, end, indices, top_k, min_score))

        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_scan_chunk_worker, *args) for args in chunk_args]
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                completed += 1
                worker_heap = fut.result()
                _merge_heaps(candidates_heap, worker_heap, top_k)
                print(f"  [{fname}] Worker chunk {completed}/{num_workers} merged ({completed/num_workers*100:.0f}%)...", flush=True)

        print(f"Finished {fname} in {time.time() - t_start:.1f}s.", flush=True)

    scan_test_pool(os.path.join(test_dir, "test_source2.tsv"))
    scan_test_pool(os.path.join(test_dir, "test_source3.tsv"))

    # 4. Write candidate_pairs.tsv
    print("\nWriting output/candidate_pairs.tsv...", flush=True)
    candidates = {s1_id: {cid for _, cid in heap} for s1_id, heap in candidates_heap.items()}
    cand_pairs_path = os.path.join(output_dir, "candidate_pairs.tsv")
    format_candidate_pairs_file(all_s1_ids, candidates, cand_pairs_path)
    print(f"Saved {cand_pairs_path} successfully.", flush=True)

    # 5. Extract candidate metadata for scoring
    needed_cand_ids = set()
    for cands in candidates.values():
        needed_cand_ids.update(cands)
    print(f"\nLoading metadata for {len(needed_cand_ids):,} candidate records for scoring...", flush=True)

    cand_records = {}
    for src in [os.path.join(test_dir, "test_source2.tsv"), os.path.join(test_dir, "test_source3.tsv")]:
        with open(src, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.strip().split("\t")
                cid = p[0]
                if cid in needed_cand_ids:
                    cn, sn, nt = normalize_business_name(p[1] if len(p) > 1 else "")
                    ca, lm, at, nums = normalize_address(p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")
                    cand_records[cid] = {
                        "clean_name": cn, "stripped_name": sn, "name_tokens": nt,
                        "clean_addr": ca, "has_landmark": 1 if lm else 0,
                        "addr_tokens": at, "num_tokens": nums, "country": p[3] if len(p) > 3 else ""
                    }
                    if len(cand_records) >= len(needed_cand_ids):
                        break

    # 6. Score candidate pairs with LightGBM in chunks
    print("\nScoring pairs with LightGBM...", flush=True)
    matches_dict = defaultdict(list)

    # Process entities in batches
    BATCH_SIZE = 50000
    all_s1_with_cands = [s1_id for s1_id in all_s1_ids if s1_id in candidates and len(candidates[s1_id]) > 0]
    
    for i in range(0, len(all_s1_with_cands), BATCH_SIZE):
        batch_s1 = all_s1_with_cands[i : i + BATCH_SIZE]
        pair_rows = []
        pair_identities = []

        for s1_id in batch_s1:
            s1 = s1_records[s1_id]
            for cid in candidates[s1_id]:
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
                pair_rows.append(feats)
                pair_identities.append((s1_id, cid))

        if pair_rows:
            batch_df = pd.DataFrame(pair_rows)[feature_cols]
            probs = model.predict_proba(batch_df)[:, 1]

            for (s1_id, cid), prob in zip(pair_identities, probs):
                if prob >= threshold:
                    matches_dict[s1_id].append((prob, cid))

    # 7. Write matching_results.tsv
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    print(f"\nWriting final matches to {matching_path}...", flush=True)

    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            if s1_id in matches_dict:
                # Sort by prob descending and cap at 8 matches
                sorted_cands = sorted(matches_dict[s1_id], key=lambda x: x[0], reverse=True)[:8]
                matched_str = ",".join(cid for _, cid in sorted_cands)
            else:
                matched_str = "" # Singleton / no confident match
            f.write(f"{s1_id}\t{matched_str}\n")

    print(f"Generated {matching_path} successfully.", flush=True)

    # 8. Run validate_submission.py
    validator_path = os.path.join(data_dir, "..", "utils", "validate_submission.py")
    if os.path.isfile(validator_path):
        print("\n" + "="*50)
        print("=== RUNNING LOCAL SUBMISSION VALIDATOR ===")
        print("="*50, flush=True)
        import subprocess
        cmd = [
            sys.executable, validator_path,
            "--matching", matching_path,
            "--candidate", cand_pairs_path,
            "--test-dir", test_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"Validator Returncode: {result.returncode} (0 = PASS)")


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_directory = os.path.join(repo_root, "dataset", "student_resource", "dataset")
    output_directory = os.path.join(repo_root, "output")
    artifacts_dir = os.path.join(repo_root, "code", "business_entity_resolution", "artifacts")
    
    m_path = os.path.join(artifacts_dir, "lgbm_model.pkl")
    cfg_path = os.path.join(artifacts_dir, "config.json")
    
    if os.path.exists(m_path) and os.path.exists(cfg_path):
        run_test_submission(m_path, cfg_path, data_directory, output_directory)
    else:
        print(f"Model artifacts not found yet at {artifacts_dir}. Train the model first via pipeline.py!")
