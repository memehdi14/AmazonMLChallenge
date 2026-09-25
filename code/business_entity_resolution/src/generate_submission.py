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
sys.path.insert(0, r"c:\MMDPublic\Hackathons\Amazon ML challenge\code\business_entity_resolution\src")
from normalize import normalize_business_name, normalize_address, get_phonetic_keys
from features import compute_pairwise_features
from blocking import format_candidate_pairs_file

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
    top_k: int = 50
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

            if st_nums:
                for sn_val in st_nums:
                    for at_val in at[:3]:
                        if len(at_val) >= 3 and not at_val.isdigit():
                            st_num_addr_index[(sn_val, at_val)].append(eid)

    print(f"Indexed {len(all_s1_ids):,} Test S1 entities in {time.time() - t0:.1f}s.", flush=True)

    # 3. Stream test_source2 and test_source3 to generate candidates
    candidates_heap = defaultdict(list)

    def scan_test_pool(filepath):
        fname = os.path.basename(filepath)
        print(f"\nScanning {fname}...", flush=True)
        t_start = time.time()
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                count += 1
                p = line.strip().split("\t")
                cid = p[0]
                name = p[1] if len(p) > 1 else ""
                addr = p[2] if len(p) > 2 else ""
                country = p[3] if len(p) > 3 else ""

                cn, sn, nt = normalize_business_name(name)
                ca, lm, at, nums = normalize_address(addr, country)
                compact_cand = "".join(nt)
                p4 = sn[:4] if len(sn) >= 4 else sn

                scores = defaultdict(float)

                if len(compact_cand) >= 4 and compact_cand in compact_name_index:
                    for s1_id in compact_name_index[compact_cand]:
                        scores[s1_id] += 12.0

                for t in nt:
                    if t in token_index:
                        for s1_id in token_index[t]:
                            scores[s1_id] += 6.0

                if p4 in prefix_index:
                    for s1_id in prefix_index[p4]:
                        scores[s1_id] += 2.0

                if nt:
                    for pk in get_phonetic_keys(nt[:1]):
                        if pk in phonetic_index:
                            for s1_id in phonetic_index[pk]:
                                scores[s1_id] += 1.5

                cand_pc = {n for n in nums if len(n) in (5, 6)}
                cand_st_nums = nums - cand_pc
                if cand_st_nums:
                    for sn_val in cand_st_nums:
                        for at_val in at[:4]:
                            key = (sn_val, at_val)
                            if key in st_num_addr_index:
                                for s1_id in st_num_addr_index[key]:
                                    scores[s1_id] += 8.0

                for s1_id, sc in scores.items():
                    if sc >= 5.0:
                        h = candidates_heap[s1_id]
                        if len(h) < top_k:
                            heapq.heappush(h, (sc, cid))
                        elif sc > h[0][0]:
                            heapq.heapreplace(h, (sc, cid))

                if count % 1000000 == 0:
                    el = time.time() - t_start
                    print(f"  [{fname}] {count:,} rows in {el:.1f}s ({count/el:,.0f} rows/s)...", flush=True)

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
    data_directory = r"c:\MMDPublic\Hackathons\Amazon ML challenge\Dataset\student_resource\dataset"
    output_directory = r"c:\MMDPublic\Hackathons\Amazon ML challenge\output"
    artifacts_dir = r"c:\MMDPublic\Hackathons\Amazon ML challenge\code\business_entity_resolution\artifacts"
    
    m_path = os.path.join(artifacts_dir, "lgbm_model.pkl")
    cfg_path = os.path.join(artifacts_dir, "config.json")
    
    if os.path.exists(m_path) and os.path.exists(cfg_path):
        run_test_submission(m_path, cfg_path, data_directory, output_directory)
    else:
        print(f"Model artifacts not found yet at {artifacts_dir}. Train the model first via pipeline.py!")
