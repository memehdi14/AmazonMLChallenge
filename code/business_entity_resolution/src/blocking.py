"""
ML Challenge 2026 - Blocking & Candidate Generation Module
Multi-channel inverted-index candidate generator supporting:
- Token-overlap blocking
- Phonetic (Soundex/Metaphone) blocking
- Character prefix blocking
- Numeric address (street number, PIN/postal code) + landmark matching
"""

import os
import heapq
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from normalize import normalize_business_name, normalize_address, get_phonetic_keys

# Shared Canonical Configuration
DEFAULT_MIN_SCORE = 3.0

COMMON_STOPWORDS = {
    # Functional / preposition words
    "the", "and", "of", "in", "at", "for", "on", "a", "an", "to", "by", "with", "from",
    # Corporate / entity types
    "co", "inc", "llc", "ltd", "corp", "corporation", "pvt", "limited", "private", "services",
    "enterprises", "company", "group", "sarl", "sas", "sa", "associates", "trading", "industries",
    # Ultra-common generic business categories
    "hotel", "restaurant", "store", "shop", "center", "centre", "solutions", "international"
}
STOPWORDS = COMMON_STOPWORDS


class CandidateGenerator:
    """
    High-recall, low-overhead candidate generator.
    Indexes target reference entities and scans candidate source pools.
    """

    def __init__(self, top_k: int = 50, min_score: float = DEFAULT_MIN_SCORE):
        self.top_k = top_k
        self.min_score = min_score

        # Indices
        self.token_index = defaultdict(list)
        self.compact_name_index = defaultdict(list)
        self.phonetic_index = defaultdict(list)
        self.prefix_index = defaultdict(list)
        self.st_num_addr_index = defaultdict(list)
        self.postal_index = defaultdict(list)

        # Entity metadata cache
        self.entity_meta = {}

    def index_reference_entities(self, records: Dict[str, Dict]):
        """
        Builds inverted indices for reference entities (e.g. Source 1 entities).
        """
        for eid, r in records.items():
            self.entity_meta[eid] = {
                "addr_tokens": set(r["addr_tokens"]),
                "num_tokens": set(r["num_tokens"]),
                "country": r.get("country", "")
            }

            nt = r["name_tokens"]
            sn = r["stripped_name"]
            nums = r["num_tokens"]
            at = r["addr_tokens"]

            # 1. Distinctive name tokens
            for t in set(nt):
                if t not in STOPWORDS and len(t) >= 3:
                    self.token_index[t].append(eid)

            # 2. Compact name (handles domain names like devotiondealmark.com)
            compact = "".join(nt)
            if len(compact) >= 4:
                self.compact_name_index[compact].append(eid)

            # 3. Phonetic keys
            pks = get_phonetic_keys(nt[:1])
            for pk in pks:
                self.phonetic_index[pk].append(eid)

            # 4. Stripped name prefix (length 4)
            prefix4 = sn[:4] if len(sn) >= 4 else sn
            if prefix4 and len(prefix4) >= 3:
                self.prefix_index[prefix4].append(eid)

            # 5. Compound Address & Postal Code Matching
            pc = {n for n in nums if len(n) in (5, 6)}
            for pc_val in pc:
                self.postal_index[pc_val].append(eid)

            st_nums = nums - pc
            if st_nums:
                for sn_val in st_nums:
                    for at_val in at:
                        if len(at_val) >= 3 and not at_val.isdigit():
                            self.st_num_addr_index[(sn_val, at_val)].append(eid)

        # High-frequency pruning step (applied once before workers receive indices)
        orig_tok = len(self.token_index)
        orig_pfx = len(self.prefix_index)
        orig_addr = len(self.st_num_addr_index)
        orig_phon = len(self.phonetic_index)
        orig_post = len(self.postal_index)

        self.token_index = {t: eids for t, eids in self.token_index.items() if len(eids) <= 300}
        self.prefix_index = {p: eids for p, eids in self.prefix_index.items() if len(eids) <= 300}
        self.st_num_addr_index = {k: eids for k, eids in self.st_num_addr_index.items() if len(eids) <= 200}
        self.phonetic_index = {pk: eids for pk, eids in self.phonetic_index.items() if len(eids) <= 300}
        self.postal_index = {pc: eids for pc, eids in self.postal_index.items() if len(eids) <= 100}

        print(f"Index pruning applied (thresholds: tok<=300, pfx<=300, addr<=200, phon<=300, post<=100):")
        print(f"  token_index      : {orig_tok:,} -> {len(self.token_index):,} keys")
        print(f"  prefix_index     : {orig_pfx:,} -> {len(self.prefix_index):,} keys")
        print(f"  st_num_addr_index: {orig_addr:,} -> {len(self.st_num_addr_index):,} keys")
        print(f"  phonetic_index   : {orig_phon:,} -> {len(self.phonetic_index):,} keys")
        print(f"  postal_index     : {orig_post:,} -> {len(self.postal_index):,} keys", flush=True)

    def scan_candidate_pool(
        self,
        filepath: str,
        candidates_heap: Dict[str, List[Tuple[float, str]]],
        num_workers: Optional[int] = None,
        max_rows: Optional[int] = None
    ):
        """
        Parallelized candidate pool scanner using ProcessPoolExecutor.
        Slices file by byte offsets across available logical CPU cores.
        """
        import pickle
        import concurrent.futures
        file_size = os.path.getsize(filepath)
        if num_workers is None:
            num_workers = max(1, min((os.cpu_count() or 4) - 2, 14))

        # 3. Print the pickled size of each index dict before dispatching to workers
        print("\n=== INDEX SERIALIZATION SIZES (PRE-DISPATCH) ===")
        total_mb = 0.0
        for name, idx in [
            ("token_index", self.token_index),
            ("compact_name_index", self.compact_name_index),
            ("phonetic_index", self.phonetic_index),
            ("prefix_index", self.prefix_index),
            ("st_num_addr_index", self.st_num_addr_index),
            ("postal_index", self.postal_index),
        ]:
            sz_mb = len(pickle.dumps(idx)) / (1024 * 1024)
            total_mb += sz_mb
            print(f"  {name:20s}: {len(idx):>8,} keys | {sz_mb:>8.2f} MB")
        print(f"  {'TOTAL PER WORKER':20s}:          | {total_mb:>8.2f} MB")
        print(f"  Estimated index RAM across {num_workers} workers: {total_mb * num_workers:>8.2f} MB ({total_mb * num_workers / 1024:.2f} GB)\n", flush=True)

        if max_rows:
            # Estimate byte offset for max_rows to avoid scanning whole file
            avg_line_bytes = 100
            scan_bytes = min(file_size, int(max_rows * avg_line_bytes))
            print(f"Truncated scan requested: max_rows={max_rows:,} (~{scan_bytes / (1024*1024):.1f} MB of {file_size / (1024*1024):.1f} MB)", flush=True)
            chunk_size = scan_bytes // num_workers
            max_rows_per_worker = max_rows // num_workers
        else:
            scan_bytes = file_size
            chunk_size = file_size // num_workers
            max_rows_per_worker = None

        print(f"Scanning {os.path.basename(filepath)} with {num_workers} parallel CPU workers (min_score={self.min_score})...", flush=True)

        chunk_args = []
        indices = (
            dict(self.token_index),
            dict(self.compact_name_index),
            dict(self.phonetic_index),
            dict(self.prefix_index),
            dict(self.st_num_addr_index),
            dict(self.postal_index)
        )

        for i in range(num_workers):
            start = i * chunk_size
            end = scan_bytes if i == num_workers - 1 else (i + 1) * chunk_size
            chunk_args.append((filepath, start, end, indices, self.top_k, self.min_score, max_rows_per_worker))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_scan_chunk_worker, *args) for args in chunk_args]
            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                completed += 1
                worker_heap = fut.result()
                _merge_heaps(candidates_heap, worker_heap, self.top_k)
                print(f"  Worker chunk {completed}/{num_workers} merged ({completed/num_workers*100:.0f}%)...", flush=True)


def _scan_chunk_worker(
    filepath: str,
    start_offset: int,
    end_offset: int,
    indices: Tuple,
    top_k: int,
    min_score: float,
    max_rows_per_worker: Optional[int] = None
) -> Dict[str, List[Tuple[float, str]]]:
    """
    Worker process that scans a specific byte slice of a candidate pool TSV.
    """
    token_index, compact_name_index, phonetic_index, prefix_index, st_num_addr_index, postal_index = indices
    local_heap = defaultdict(list)
    row_count = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start_offset)
        if start_offset != 0:
            f.readline()  # Skip incomplete partial line
        else:
            f.readline()  # Skip TSV header line

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

            # Channel A: Compact Name Exact Match (exact match, max 15)
            if len(compact_cand) >= 4 and compact_cand in compact_name_index:
                for ref_id in compact_name_index[compact_cand][:15]:
                    matched_scores[ref_id] += 12.0

            # Channel B: Distinctive name tokens (bounded to 15 per token)
            for t in cand_nt_set:
                if t in token_index:
                    for ref_id in token_index[t][:15]:
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

            # Channel E: Compound Address Match (bounded to 15)
            cand_pc = {n for n in cand_num_set if len(n) in (5, 6)}
            cand_st_nums = cand_num_set - cand_pc
            if cand_st_nums:
                for sn_val in cand_st_nums:
                    for at_val in addr_tokens:
                        key = (sn_val, at_val)
                        if key in st_num_addr_index:
                            for ref_id in st_num_addr_index[key][:15]:
                                matched_scores[ref_id] += 8.0

            # Channel F: Postal Code Exact Match (bounded to 10)
            for cand_pc_val in cand_pc:
                if cand_pc_val in postal_index:
                    for ref_id in postal_index[cand_pc_val][:10]:
                        matched_scores[ref_id] += 10.0

            # Push to local heaps
            for ref_id, score in matched_scores.items():
                if score >= min_score:
                    h = local_heap[ref_id]
                    if len(h) < top_k:
                        heapq.heappush(h, (score, cand_id))
                    elif score > h[0][0]:
                        heapq.heapreplace(h, (score, cand_id))

    return dict(local_heap)


def _merge_heaps(
    target_heap: Dict[str, List[Tuple[float, str]]],
    source_heap: Dict[str, List[Tuple[float, str]]],
    top_k: int
):
    """Merges a worker heap into the master candidates heap."""
    for ref_id, items in source_heap.items():
        h = target_heap[ref_id]
        for sc, cid in items:
            if len(h) < top_k:
                heapq.heappush(h, (sc, cid))
            elif sc > h[0][0]:
                heapq.heapreplace(h, (sc, cid))


def format_candidate_pairs_file(
    all_s1_ids: List[str],
    candidates: Dict[str, Set[str]],
    output_path: str
):
    """
    Writes candidates to candidate_pairs.tsv matching competition format:
    source1_entity_id \t candidate_entity_ids (comma-separated, empty for singletons)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = candidates.get(s1_id, set())
            cand_str = ",".join(sorted(cands)) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
