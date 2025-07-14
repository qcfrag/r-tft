from datetime import datetime
import pathlib
import re
from typing import Any, Union
import numpy as np

# Date when forbidden lists were last updated
ethics_update_date = datetime(2025, 7, 14)

BLOCK_DIR = pathlib.Path(__file__).with_suffix('').parent

def _load_list(fname):
    path = BLOCK_DIR / fname
    if not path.exists():
        return set()
    raw = path.read_text(encoding="utf8").splitlines()
    return {line.strip().lower() for line in raw if line.strip()}

FORBIDDEN = (
    _load_list("forbidden_domains.txt")
    | _load_list("forbidden_companies.txt")
    | _load_list("forbidden_keywords.txt")
)
ALLOWED_DOMAINS = set(_load_list("allowed_domains.txt"))

# Build optimized regex pattern for forbidden terms
FORBIDDEN_REGEX = re.compile(rf"\b({'|'.join(map(re.escape, FORBIDDEN))})\b") if FORBIDDEN else None

# Homoglyph defense mapping
HOMOGLYPH_MAP = str.maketrans({
    '\u0430': 'a',  # Cyrillic 'а'
    '\u0435': 'e',  # Cyrillic 'е'
    '\u0456': 'i',  # Cyrillic 'і'
    '\u043E': 'o',  # Cyrillic 'о'
    '\u0440': 'p',  # Cyrillic 'р'
    '\u0455': 's',  # Cyrillic 'ѕ'
    '\u0501': 'd',  # Cyrillic 'ԁ'
    '\u051B': 'h',  # Cyrillic 'һ'
    '\u1EA1': 'a',  # Latin 'ạ'
    '\u1EB9': 'e',  # Latin 'ẹ'
    '\u00E8': 'e',  # Latin 'è'
    # Add more mappings as needed
})

class ResonanceEthicsError(Exception):
    pass

def normalize_homoglyphs(text: str) -> str:
    """Defend against homoglyph attacks"""
    return text.translate(HOMOGLYPH_MAP)

def _deep_meta_check(value: Any) -> None:
    """Recursively check nested metadata structures"""
    if isinstance(value, dict):
        for v in value.values():
            _deep_meta_check(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _deep_meta_check(item)
    else:
        sval = str(value)
        # Defense against hex/unicode evasion
        if '\\x' in sval or '\\u' in sval:
            try:
                sval = bytes(sval, 'utf-8').decode('unicode_escape')
            except:
                pass
        
        sval = normalize_homoglyphs(sval).lower()
        
        # Check against forbidden patterns
        if FORBIDDEN_REGEX and FORBIDDEN_REGEX.search(sval):
            raise ResonanceEthicsError(
                f"Metadata violation: forbidden pattern detected in value"
            )

def check_domain(domain: str):
    d = (domain or "").lower()
    d = normalize_homoglyphs(d)
    
    # Explicit whitelist check
    if d in ALLOWED_DOMAINS:
        return
        
    # Optimized regex check for forbidden terms
    if FORBIDDEN_REGEX and FORBIDDEN_REGEX.search(d):
        raise ResonanceEthicsError(
            f"REL-1.0 violation: forbidden pattern detected in domain '{domain}'"
            )
            
    # Contextual analysis for ambiguous terms
    ambiguous_terms = {
        "resonance": "resonance_research" in ALLOWED_DOMAINS,
        "phase": "phase_detection" in ALLOWED_DOMAINS,
        "quantum": "quantum_computing" in ALLOWED_DOMAINS
    }
    
    for term, is_allowed in ambiguous_terms.items():
        if term in d:
            if is_allowed:
                return  # Benign usage in allowed context
            raise ResonanceEthicsError(
                f"Ambiguous term '{term}' in non-allowed context: '{domain}'"
            )
            
    # If not explicitly allowed, treat as disallowed
    raise ResonanceEthicsError(
        f"REL-1.0 violation: domain '{domain}' is not in allowed list"
    )

def check_metadata(meta: dict[str, Any] | None):
    if not meta:
        return
    _deep_meta_check(meta)

# Contextual analysis stubs (for future implementation)
def contextual_analysis(domain: str, vector: np.ndarray) -> bool:
    """Future: ML-powered context analysis"""
    # Placeholder for:
    # 1. Vector pattern analysis
    # 2. Semantic context evaluation
    # 3. Dual-use detection
    return True  # Always passes for now

def detect_fragmentation_attack(text: str) -> bool:
    """Detect fragmented forbidden terms"""
    # Placeholder for:
    # 1. Term recombination analysis
    # 2. Steganography detection
    # 3. Semantic segmentation defense
    return False  # No detection for now
