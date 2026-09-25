"""
Profile single-worker chunk scanning on test_source2.tsv using cProfile.
Supports standard scan vs Step 4 bounded per-channel top-N scan.
Computes estimated candidate_pairs.tsv file size vs 512MB hard limit (Instruction 5).
"""

import os
import sys
import time
import heapq
import cProfile
import pstats
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_business_name, normalize_address, get_phonetic_keys
from blocking import _scan_chunk_worker, COMMON_STOPWORDS, DEFAULT_MIN_SCORE


def _scan_chunk_worker_step4(
    filepath: str,
    start_offset: int,
    end_offset: int,
    indices: tuple,
    top_k: int = 20,
    min_score: float = 8.0,
    max_rows_per_worker: Optional[int] = None,
    chan_k: int = 15
) -> Dict[str, List[Tuple[float, str]]]:
    """
    Step 4: Per-channel bounded top-N scanning.
    Bounds candidate list lookups per channel to avoid combinatorial posted-list explosion.
    """
    token_index, compact_name_index, phonetic_index, prefix_index, st_num_addr_index, postal_index = indices
    local_heap = defaultdict(list)
    row_count = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start_offset)
        if start_offset != 0:
            f.readline()
        else:
            f.readline()

        while True:
            if max_rows_per_worker and row_count >= max_rows_per_worker:
                break
            if end_offset > 0 and f.tell() >= end_offset:
                break
            line = f.readline()
            if not line:
                break
            row_count += 1

            parts = line.strip().split("\t")
            cand_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""

            clean_name, stripped_name, name_tokens = normalize_business_name(name)
            clean_addr, landmark, addr_tokens, num_tokens = normalize_address(addr, country)

            cand_nt_set = set(name_tokens)
            cand_num_set = set(num_tokens)
            cand_pks = get_phonetic_keys(name_tokens[:1])
            cand_prefix = stripped_name[:4] if len(stripped_name) >= 4 else stripped_name
            compact_cand = "".join(name_tokens)

            matched_scores = defaultdict(float)

            # Channel A: Compact Name Exact Match (exact match, max chan_k)
            if len(compact_cand) >= 4 and compact_cand in compact_name_index:
                for ref_id in compact_name_index[compact_cand][:chan_k]:
                    matched_scores[ref_id] += 12.0

            # Channel B: Distinctive name tokens (bounded to chan_k per token)
            for t in cand_nt_set:
                if t in token_index:
                    for ref_id in token_index[t][:chan_k]:
                        matched_scores[ref_id] += 6.0

            # Channel C: Stripped 4-prefix (bounded to 10)
            if cand_prefix in prefix_index:
                for ref_id in prefix_index[cand_prefix][:10]:
                    matched_scores[ref_id] += 2.0

            # Channel D: Phonetic similarity (bounded to 10)
            for pk in cand_pks:
                if pk in phonetic_index:
                    for ref_id in phonetic_index[pk][:10]:
                        matched_scores[ref_id] += 1.5

            # Channel E: Compound Address Match (bounded to chan_k)
            cand_pc = {n for n in cand_num_set if len(n) in (5, 6)}
            cand_st_nums = cand_num_set - cand_pc
            if cand_st_nums:
                for sn_val in cand_st_nums:
                    for at_val in addr_tokens:
                        key = (sn_val, at_val)
                        if key in st_num_addr_index:
                            for ref_id in st_num_addr_index[key][:chan_k]:
                                matched_scores[ref_id] += 8.0

            # Channel F: Postal Code Match (bounded to 10)
            for cand_pc_val in cand_pc:
                if cand_pc_val in postal_index:
                    for ref_id in postal_index[cand_pc_val][:10]:
                        matched_scores[ref_id] += 10.0

            # Push to local heaps bounded by top_k
            for ref_id, score in matched_scores.items():
                if score >= min_score:
                    h = local_heap[ref_id]
                    if len(h) < top_k:
                        heapq.heappush(h, (score, cand_id))
                    elif score > h[0][0]:
                        heapq.heapreplace(h, (score, cand_id))

    return dict(local_heap)


def build_test_indices(s1_path: str, max_s1: int = None):
    print(f"Indexing reference entities from {s1_path}...", flush=True)
    t0 = time.time()
    token_index = defaultdict(list)
    compact_name_index = defaultdict(list)
    phonetic_index = defaultdict(list)
    prefix_index = defaultdict(list)
    st_num_addr_index = defaultdict(list)
    postal_index = defaultdict(list)

    count = 0
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline() # Header
        for line in f:
            count += 1
            if max_s1 and count > max_s1:
                break
            parts = line.strip().split("\t")
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""

            cn, sn, nt = normalize_business_name(name)
            ca, lm, at, nums = normalize_address(addr, country)
            pks = get_phonetic_keys(nt[:1])
            compact = "".join(nt)
            p4 = sn[:4] if len(sn) >= 4 else sn
            pc = {n for n in nums if len(n) in (5, 6)}
            st_nums = nums - pc

            for t in set(nt):
                if t not in COMMON_STOPWORDS and len(t) >= 3:
                    token_index[t].append(eid)
            if len(compact) >= 4:
                compact_name_index[compact].append(eid)
            for pk in pks:
                phonetic_index[pk].append(eid)
            if p4 and len(p4) >= 3:
                prefix_index[p4].append(eid)
            for pc_val in pc:
                postal_index[pc_val].append(eid)
            if st_nums:
                for sn_val in st_nums:
                    for at_val in at:
                        if len(at_val) >= 3 and not at_val.isdigit():
                            st_num_addr_index[(sn_val, at_val)].append(eid)

    print(f"Loaded {count:,} S1 entities in {time.time() - t0:.1f}s.", flush=True)

    # Standard pruning
    token_index = {t: eids for t, eids in token_index.items() if len(eids) <= 300}
    prefix_index = {p: eids for p, eids in prefix_index.items() if len(eids) <= 300}
    st_num_addr_index = {k: eids for k, eids in st_num_addr_index.items() if len(eids) <= 200}
    phonetic_index = {pk: eids for pk, eids in phonetic_index.items() if len(eids) <= 300}
    postal_index = {pc: eids for pc, eids in postal_index.items() if len(eids) <= 100}

    print("Index pruning applied.", flush=True)
    return (token_index, compact_name_index, phonetic_index, prefix_index, st_num_addr_index, postal_index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rows", type=int, default=30000, help="Number of candidate rows to scan under cProfile")
    parser.add_argument("--min-score", type=float, default=8.0, help="Candidate min_score threshold (default: 8.0)")
    parser.add_argument("--top-k", type=int, default=20, help="Candidate heap cap (default: 20)")
    parser.add_argument("--step4", action="store_true", help="Use Step 4 per-channel bounded scanning")
    parser.add_argument("--max-s1", type=int, default=None, help="Limit S1 indexing (default: all)")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    s1_path = os.path.join(repo_root, "dataset", "student_resource", "dataset", "test", "test_source1.tsv")
    s2_path = os.path.join(repo_root, "dataset", "student_resource", "dataset", "test", "test_source2.tsv")

    indices = build_test_indices(s1_path, max_s1=args.max_s1)

    scan_fn = _scan_chunk_worker_step4 if args.step4 else _scan_chunk_worker
    scan_name = "Step 4 (Per-Channel Bounded)" if args.step4 else "Baseline (Unbounded)"

    print(f"\nProfiling {scan_name} on {args.num_rows:,} rows of {os.path.basename(s2_path)} (min_score={args.min_score}, top_k={args.top_k})...", flush=True)
    profile_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_scan.pstats")

    profiler = cProfile.Profile()
    t0 = time.time()
    profiler.enable()

    heap_res = scan_fn(
        filepath=s2_path,
        start_offset=0,
        end_offset=0,
        indices=indices,
        top_k=args.top_k,
        min_score=args.min_score,
        max_rows_per_worker=args.num_rows
    )

    profiler.disable()
    dt = time.time() - t0
    profiler.dump_stats(profile_out)

    stats = pstats.Stats(profiler)
    heappush_calls = 0
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        if "heappush" in str(func):
            heappush_calls += nc

    total_candidates = sum(len(h) for h in heap_res.values())
    touched_entities = len(heap_res)
    avg_cands = total_candidates / touched_entities if touched_entities > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"PROFILE RUN RESULTS ({scan_name}, min_score={args.min_score:.1f}, top_k={args.top_k}, rows={args.num_rows:,}):")
    print(f"  Throughput:              {args.num_rows / dt:,.0f} rows/sec (took {dt:.2f}s)")
    print(f"  Total heap pushes:       {heappush_calls:,}")
    print(f"  Touched S1 entities:     {touched_entities:,}")
    print(f"  Total candidate entries: {total_candidates:,}")
    print(f"  Avg cands/entity:        {avg_cands:.2f}")
    print("=" * 80)

    # Instruction 5: Candidate Pairs File Size Estimation
    NUM_S1_TEST = 1732544
    estimated_bytes = NUM_S1_TEST * avg_cands * 11
    estimated_mb = estimated_bytes / (1024 * 1024)
    max_mb = 512.0
    safety_margin = 0.20
    ceiling_mb = max_mb * (1.0 - safety_margin) # 409.6 MB

    print("\n" + "=" * 80)
    print("INSTRUCTION 5: ESTIMATED candidate_pairs.tsv SIZE CHECK")
    print(f"  Formula: {NUM_S1_TEST:,} entities * {avg_cands:.2f} avg_cands * ~11 bytes")
    print(f"  Estimated Size:          {estimated_mb:.1f} MB ({estimated_bytes:,.0f} bytes)")
    print(f"  Hard Platform Limit:     {max_mb:.1f} MB")
    print(f"  Safe Limit (-20% margin): {ceiling_mb:.1f} MB")

    if estimated_mb >= ceiling_mb:
        print(f"  STATUS: [FAIL] Estimated size ({estimated_mb:.1f} MB) is within 20% of 512 MB ceiling! Requires stricter top_k / min_score.")
    else:
        margin_pct = (1.0 - (estimated_mb / max_mb)) * 100
        print(f"  STATUS: [PASS] Estimated size ({estimated_mb:.1f} MB) is safely below 512 MB (Margin: {margin_pct:.1f}% headroom).")
    print("=" * 80)

    print("\nTOP 15 FUNCTIONS BY CUMULATIVE TIME:")
    p_cum = pstats.Stats(profiler)
    p_cum.strip_dirs().sort_stats("cumulative").print_stats(15)

    print("\nTOP 15 FUNCTIONS BY SELF (TOTTIME) TIME:")
    p_tot = pstats.Stats(profiler)
    p_tot.strip_dirs().sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
