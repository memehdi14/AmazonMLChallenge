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

STOPWORDS = {
    "the", "and", "of", "in", "at", "for", "on", "a", "an", "to", "by", "with", "from",
    "co", "inc", "llc", "ltd", "corp", "corporation", "pvt", "limited", "private", "services",
    "enterprises", "company", "group", "sarl", "sas", "sa"
}


class CandidateGenerator:
    """
    High-recall, low-overhead candidate generator.
    Indexes target reference entities and scans candidate source pools.
    """

    def __init__(self, top_k: int = 50, min_score: float = 3.0):
        self.top_k = top_k
        self.min_score = min_score

        # Indices
        self.token_index = defaultdict(list)
        self.phonetic_index = defaultdict(list)
        self.prefix_index = defaultdict(list)
        self.addr_num_index = defaultdict(list)

        # Entity metadata cache
        self.entity_meta = {}

    def index_reference_entities(self, records: Dict[str, Dict]):
        """
        Builds inverted indices for reference entities (e.g. Source 1 entities).
        records mapping: entity_id -> {
            'clean_name': str,
            'stripped_name': str,
            'name_tokens': List[str],
            'clean_addr': str,
            'addr_tokens': List[str],
            'num_tokens': Set[str],
            'country': str
        }
        """
        for eid, r in records.items():
            self.entity_meta[eid] = {
                "addr_tokens": set(r["addr_tokens"]),
                "num_tokens": set(r["num_tokens"]),
                "country": r.get("country", "")
            }

            # 1. Distinctive name tokens
            for t in set(r["name_tokens"]):
                if t not in STOPWORDS and len(t) >= 3:
                    self.token_index[t].append(eid)

            # 2. Phonetic keys
            pks = get_phonetic_keys(r["name_tokens"][:2])
            for pk in pks:
                self.phonetic_index[pk].append(eid)

            # 3. Stripped name prefix (length 4)
            prefix4 = r["stripped_name"][:4] if len(r["stripped_name"]) >= 4 else r["stripped_name"]
            if prefix4 and len(prefix4) >= 3:
                self.prefix_index[prefix4].append(eid)

            # 4. Numeric address tokens
            for nt in r["num_tokens"]:
                if len(nt) >= 2:
                    self.addr_num_index[nt].append(eid)

    def scan_candidate_pool(
        self,
        filepath: str,
        candidates_heap: Dict[str, List[Tuple[float, str]]],
        max_rows: Optional[int] = None,
        callback_interval: int = 1000000
    ):
        """
        Streams through a candidate pool TSV (Source 2 or Source 3)
        and scores plausible matches against indexed reference entities.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            next(f) # header
            count = 0
            for line in f:
                count += 1
                parts = line.strip().split("\t")
                cand_id = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                country = parts[3] if len(parts) > 3 else ""

                clean_name, stripped_name, name_tokens = normalize_business_name(name)
                clean_addr, landmark, addr_tokens, num_tokens = normalize_address(addr, country)

                cand_nt_set = set(name_tokens)
                cand_at_set = set(addr_tokens)
                cand_num_set = set(num_tokens)
                cand_pks = get_phonetic_keys(name_tokens[:2])
                cand_prefix = stripped_name[:4] if len(stripped_name) >= 4 else stripped_name

                # Score potential references
                matched_scores = defaultdict(float)

                # Channel A: Distinctive name tokens
                for t in cand_nt_set:
                    if t in self.token_index:
                        for ref_id in self.token_index[t]:
                            matched_scores[ref_id] += 3.0

                # Channel B: Stripped 4-prefix
                if cand_prefix in self.prefix_index:
                    for ref_id in self.prefix_index[cand_prefix]:
                        matched_scores[ref_id] += 2.0

                # Channel C: Phonetic similarity
                for pk in cand_pks:
                    if pk in self.phonetic_index:
                        for ref_id in self.phonetic_index[pk]:
                            matched_scores[ref_id] += 1.5

                # Channel D: Numeric address matching
                for nt in cand_num_set:
                    if nt in self.addr_num_index:
                        for ref_id in self.addr_num_index[nt]:
                            if self.entity_meta[ref_id]["addr_tokens"].intersection(cand_at_set):
                                matched_scores[ref_id] += 4.0

                # Push to heaps
                for ref_id, score in matched_scores.items():
                    if score >= self.min_score:
                        heap = candidates_heap[ref_id]
                        if len(heap) < self.top_k:
                            heapq.heappush(heap, (score, cand_id))
                        elif score > heap[0][0]:
                            heapq.heapreplace(heap, (score, cand_id))

                if max_rows and count >= max_rows:
                    break


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
