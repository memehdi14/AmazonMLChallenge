"""
ML Challenge 2026 - Data Normalization Module
Handles multilingual text normalization, legal suffix stripping,
country-aware address expansion, landmark extraction, and phonetic hashing.
"""

import re
import unicodedata
from typing import Tuple, List, Set, Optional
import jellyfish

# Legal suffixes across US, India, and France
LEGAL_SUFFIX_PATTERNS = [
    # Multi-word suffixes first
    r"\bprivate limited\b",
    r"\bpvt limited\b",
    r"\bpvt ltd\b",
    r"\bpvt\b",
    r"\bcorporation\b",
    r"\bincorporated\b",
    r"\blimited liability company\b",
    r"\blimited liability partnership\b",
    r"\bholding company\b",
    # Transliterated suffixes
    r"\bpraivet limited\b",
    r"\bpraivet\b",
    r"\belelpi\b",
    # Single-word suffixes
    r"\bcorp\b",
    r"\binc\b",
    r"\bllc\b",
    r"\bltd\b",
    r"\bllp\b",
    r"\bco\b",
    r"\bcompany\b",
    r"\bgroup\b",
    r"\benterprises\b",
    r"\bservices\b",
    # French suffixes
    r"\bsarl\b",
    r"\bsas\b",
    r"\bsasu\b",
    r"\bsa\b",
    r"\beurl\b",
    r"\bsci\b",
    r"\bsnc\b",
    r"\bgie\b",
]

_LEGAL_SUFFIX_REGEX = re.compile("|".join(LEGAL_SUFFIX_PATTERNS), flags=re.IGNORECASE)

# Landmark patterns in address (especially frequent in India)
LANDMARK_PATTERN = re.compile(
    r"\b(near|opp|opposite|behind|beside|next to|nr|adj|adjacent to|in front of|close to|opp to)\b\s+([^,]+)",
    flags=re.IGNORECASE
)

# Address abbreviations by country / general
ADDRESS_ABBR = {
    # US & General
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "dr": "drive",
    "blvd": "boulevard",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "hwy": "highway",
    "pkwy": "parkway",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "sq": "square",
    "ctr": "center",
    "tpke": "turnpike",
    # India specific
    "nr": "near",
    "opp": "opposite",
    "soc": "society",
    "apt": "apartment",
    "mkt": "market",
    "ext": "extension",
    # France specific
    "bd": "boulevard",
    "av": "avenue",
    "all": "allee",
    "imp": "impasse",
    "r": "rue",
    "rte": "route",
    "pl": "place",
    "faub": "faubourg",
}


import anyascii

WEB_SUFFIX_REGEX = re.compile(r"\b(www\.|https?://|http://)?([a-z0-9\-]+)\.(com|in|org|net|co|io|biz|info|fr)\b", flags=re.IGNORECASE)


def strip_accents(text: str) -> str:
    """Normalize unicode characters (NFKD) and strip accents (e.g., é -> e)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_base_text(text: str) -> str:
    """
    Base text cleaning:
    - Transliterate non-Latin scripts (Hindi, Tamil, Telugu, etc.) to Latin via anyascii
    - Lowercase, accent fold, strip web domains, & -> and, strip punctuation, collapse whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    # Transliterate any non-Latin scripts to ASCII
    text = anyascii.anyascii(text)
    # Strip accents
    text = strip_accents(text.lower())
    # Handle web domains: "maurewilliamscolombier.com" -> "maurewilliamscolombier"
    text = WEB_SUFFIX_REGEX.sub(r"\2", text)
    text = text.replace("&", " and ")
    # Replace non-alphanumeric with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    return " ".join(text.split())


def normalize_business_name(name: str) -> Tuple[str, str, List[str]]:
    """
    Returns:
    - clean_name: normalized name
    - stripped_name: name with legal suffixes removed
    - tokens: list of meaningful tokens
    """
    clean = clean_base_text(name)
    # Remove legal suffixes
    stripped = _LEGAL_SUFFIX_REGEX.sub(" ", clean)
    stripped = " ".join(stripped.split())
    if not stripped:
        stripped = clean  # Fallback if name was only a suffix

    tokens = [t for t in stripped.split() if len(t) > 1]
    return clean, stripped, tokens


def extract_landmarks(address: str) -> Tuple[str, str]:
    """
    Extracts landmark phrases into a separate string,
    returning (clean_address_without_landmark, landmark_text).
    """
    if not address or not isinstance(address, str):
        return "", ""

    matches = LANDMARK_PATTERN.findall(address)
    landmarks = " ".join(f"{m[0]} {m[1].strip()}" for m in matches)
    cleaned = LANDMARK_PATTERN.sub(" ", address)
    return cleaned, landmarks


def normalize_address(address: str, country: Optional[str] = None) -> Tuple[str, str, List[str], Set[str]]:
    """
    Normalizes address text, expands abbreviations, extracts landmarks and numeric tokens.
    
    Returns:
    - clean_addr: normalized expanded address
    - landmark: extracted landmark string
    - tokens: word tokens
    - numeric_tokens: set of numbers (street numbers, zip/pin codes)
    """
    cleaned_raw, landmark = extract_landmarks(address)
    clean = clean_base_text(cleaned_raw)

    words = clean.split()
    expanded_words = [ADDRESS_ABBR.get(w, w) for w in words]
    clean_addr = " ".join(expanded_words)

    tokens = [w for w in expanded_words if len(w) > 1]
    numeric_tokens = set(re.findall(r"\b\d+\b", clean))

    clean_landmark = clean_base_text(landmark)

    return clean_addr, clean_landmark, tokens, numeric_tokens


def get_phonetic_keys(tokens: List[str]) -> List[str]:
    """Generates Soundex and Metaphone keys for tokens."""
    keys = []
    for t in tokens[:3]: # First 3 tokens of name are most distinctive
        if t.isalpha() and len(t) >= 2:
            try:
                keys.append(jellyfish.soundex(t))
                m = jellyfish.metaphone(t)
                if m:
                    keys.append(m)
            except Exception:
                pass
    return list(set(keys))
