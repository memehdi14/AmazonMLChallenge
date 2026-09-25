"""
ML Challenge 2026 - Pairwise Feature Engineering Module
Computes discriminative name, address, and meta features for candidate pairs
using ultra-fast C++ implementations from RapidFuzz.
"""

import re
from typing import Dict, Any, List, Set
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def extract_acronym(tokens: List[str]) -> str:
    """Extract first letters of non-empty tokens."""
    return "".join(t[0] for t in tokens if t).lower()


def compute_pairwise_features(
    s1_name_clean: str,
    s1_name_stripped: str,
    s1_name_tokens: List[str],
    s1_addr_clean: str,
    s1_addr_tokens: List[str],
    s1_num_tokens: Set[str],
    s1_country: str,
    s1_has_landmark: int,
    cand_id: str,
    cand_name_clean: str,
    cand_name_stripped: str,
    cand_name_tokens: List[str],
    cand_addr_clean: str,
    cand_addr_tokens: List[str],
    cand_num_tokens: Set[str],
    cand_country: str,
    cand_has_landmark: int,
) -> Dict[str, float]:
    """
    Computes a vector of pairwise features between a Source 1 entity and a candidate.
    """
    feats = {}

    # 1. Source Indicator
    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    feats["is_source2"] = is_s2
    feats["is_source3"] = 1.0 - is_s2

    # 2. Country Match (Soft Feature)
    feats["country_match"] = 1.0 if s1_country == cand_country else 0.0

    # 3. Name Similarities
    # Normalized Levenshtein (0 to 1)
    feats["name_lev_sim"] = Levenshtein.normalized_similarity(s1_name_clean, cand_name_clean)
    feats["name_stripped_lev_sim"] = Levenshtein.normalized_similarity(s1_name_stripped, cand_name_stripped)

    # Jaro-Winkler (rewards prefix matches heavily)
    feats["name_jaro_winkler"] = JaroWinkler.similarity(s1_name_clean, cand_name_clean)
    feats["name_stripped_jw"] = JaroWinkler.similarity(s1_name_stripped, cand_name_stripped)

    # RapidFuzz Ratios (scaled 0.0 to 1.0)
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(s1_name_clean, cand_name_clean) / 100.0
    feats["name_token_set_ratio"] = fuzz.token_set_ratio(s1_name_clean, cand_name_clean) / 100.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(s1_name_stripped, cand_name_stripped) / 100.0

    # Token Jaccard
    s1_nt_set = set(s1_name_tokens)
    cand_nt_set = set(cand_name_tokens)
    feats["name_token_jaccard"] = jaccard_similarity(s1_nt_set, cand_nt_set)
    feats["name_token_overlap_count"] = float(len(s1_nt_set.intersection(cand_nt_set)))

    # Character 3-gram Jaccard
    s1_ngrams = get_char_ngrams(s1_name_stripped, n=3)
    cand_ngrams = get_char_ngrams(cand_name_stripped, n=3)
    feats["name_char_3gram_jaccard"] = jaccard_similarity(s1_ngrams, cand_ngrams)

    # Exact match flags
    feats["name_exact_match"] = 1.0 if s1_name_clean == cand_name_clean else 0.0
    feats["name_stripped_exact"] = 1.0 if s1_name_stripped == cand_name_stripped else 0.0

    # Acronym match
    s1_acronym = extract_acronym(s1_name_tokens)
    cand_acronym = extract_acronym(cand_name_tokens)
    feats["name_acronym_match"] = 1.0 if (len(s1_acronym) >= 2 and s1_acronym == cand_acronym) else 0.0
    feats["name_acronym_cross_match"] = 1.0 if (
        (len(s1_acronym) >= 2 and s1_acronym == cand_name_stripped) or
        (len(cand_acronym) >= 2 and cand_acronym == s1_name_stripped)
    ) else 0.0

    # Name Length metrics
    len_s1 = max(len(s1_name_stripped), 1)
    len_cand = max(len(cand_name_stripped), 1)
    feats["name_len_diff"] = abs(len_s1 - len_cand)
    feats["name_len_ratio"] = min(len_s1, len_cand) / max(len_s1, len_cand)

    # 4. Address Similarities
    feats["addr_lev_sim"] = Levenshtein.normalized_similarity(s1_addr_clean, cand_addr_clean)
    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(s1_addr_clean, cand_addr_clean) / 100.0
    feats["addr_token_set_ratio"] = fuzz.token_set_ratio(s1_addr_clean, cand_addr_clean) / 100.0

    s1_at_set = set(s1_addr_tokens)
    cand_at_set = set(cand_addr_tokens)
    feats["addr_token_jaccard"] = jaccard_similarity(s1_at_set, cand_at_set)
    feats["addr_token_overlap_count"] = float(len(s1_at_set.intersection(cand_at_set)))

    # Numeric tokens (Street numbers, PIN codes)
    feats["num_token_jaccard"] = jaccard_similarity(s1_num_tokens, cand_num_tokens)
    feats["num_token_overlap_count"] = float(len(s1_num_tokens.intersection(cand_num_tokens)))
    feats["has_common_num_token"] = 1.0 if (s1_num_tokens and cand_num_tokens and bool(s1_num_tokens.intersection(cand_num_tokens))) else 0.0

    # Landmark features
    feats["s1_has_landmark"] = float(s1_has_landmark)
    feats["cand_has_landmark"] = float(cand_has_landmark)
    feats["both_have_landmark"] = 1.0 if (s1_has_landmark and cand_has_landmark) else 0.0

    # Combined composite similarity
    feats["composite_name_addr_sim"] = 0.65 * feats["name_token_set_ratio"] + 0.35 * feats["addr_token_set_ratio"]

    return feats
