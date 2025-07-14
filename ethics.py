from datetime import datetime
import pathlib
import re
from typing import Any
import numpy as np
from dateutil.relativedelta import relativedelta

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
    '\u0259': 'e',  # Schwa
    '\u00F6': 'o',  # ö
    '\u00FC': 'u',  # ü
})

# Weaponization pattern (placeholder values - should be trained model in production)
WEAPON_PATTERN = np.array([
    0.78, -0.12, 0.05, 1.23, -0.45, 0.89, -1.56, 0.32,
    -0.91, 0.67, 1.45, -0.23, 0.58, -1.21, 0.76, 0.34
])

class ResonanceEthicsError(Exception):
    pass

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Calculate cosine similarity between two vectors"""
    denom = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(np.dot(a, b) / denom)

def normalize_homoglyphs(text: str) -> str:
    """Defend against homoglyph attacks"""
    return text.translate(HOMOGLYPH_MAP)

def detect_term_fragmentation(text: str) -> bool:
    """Detect split forbidden terms in text"""
    fragments = {
        "weap": "on", "bra": "inwash", "neuro": "hack", 
        "reson": "ance_weapon", "quant": "um_warfare",
        "phase": "_weapon", "tempor": "al_weapon",
        "psych": "otronic", "info": "rmation_warfare"
    }
    normalized = normalize_homoglyphs(text.lower())
    
    for prefix, suffix in fragments.items():
        if prefix in normalized and suffix in normalized:
            if normalized.index(prefix) < normalized.index(suffix):
                return True
    return False

def detect_vector_weaponization(vector: np.ndarray) -> bool:
    """Detect weaponization patterns in vectors"""
    if len(vector) < len(WEAPON_PATTERN):
        return False
        
    # Pad or truncate vector to match pattern length
    vec = np.resize(vector, WEAPON_PATTERN.shape)
    
    if cosine_sim(vec, WEAPON_PATTERN) > 0.85:
        return True
        
    # Check for high-dimensional coercion signatures
    if np.max(vector) > 0.95 and np.min(vector) < -0.95:
        return True
        
    # Temporal manipulation signature
    if np.mean(vector[:4]) > 0.9 and np.mean(vector[4:8]) < -0.9:
        return True
        
    return False

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
        
        # Term fragmentation detection
        if detect_term_fragmentation(sval):
            raise ResonanceEthicsError("Term fragmentation attack detected")
        
        # Check against forbidden patterns
        if FORBIDDEN_REGEX and FORBIDDEN_REGEX.search(sval):
            raise ResonanceEthicsError("Forbidden pattern detected")

def check_domain(domain: str):
    if not domain:
        raise ResonanceEthicsError("Domain cannot be empty")
        
    d = domain.lower()
    d = normalize_homoglyphs(d)
    
    # Explicit whitelist check
    if d in ALLOWED_DOMAINS:
        return
        
    # Term fragmentation detection
    if detect_term_fragmentation(d):
        raise ResonanceEthicsError(f"Domain fragmentation attack: '{domain}'")
        
    # Optimized regex check for forbidden terms
    if FORBIDDEN_REGEX and FORBIDDEN_REGEX.search(d):
        raise ResonanceEthicsError(f"Forbidden pattern in domain: '{domain}'")
            
    # Contextual analysis for ambiguous terms
    ambiguous_terms = {
        "resonance": "resonance_research" in ALLOWED_DOMAINS,
        "phase": "phase_detection" in ALLOWED_DOMAINS,
        "quantum": "quantum_computing" in ALLOWED_DOMAINS,
        "temporal": "temporal_paradox_prevention" in ALLOWED_DOMAINS,
        "neuro": "neuroscience" in ALLOWED_DOMAINS
    }
    
    for term, is_allowed in ambiguous_terms.items():
        if term in d:
            if is_allowed:
                return  # Benign usage in allowed context
            raise ResonanceEthicsError(f"Ambiguous term '{term}' in domain: '{domain}'")
            
    # If not explicitly allowed, treat as disallowed
    raise ResonanceEthicsError(f"Domain not allowed: '{domain}'")

def check_metadata(meta: dict[str, Any] | None):
    if meta is None:
        return
    _deep_meta_check(meta)
    
def temporal_integrity_check(date: datetime):
    """Check for temporal anomalies"""
    now = datetime.now()
    if date > now + relativedelta(minutes=5):  # Allow small clock drift
        raise ResonanceEthicsError("Future-dated record detected")
    if date < now - relativedelta(years=10):
        raise ResonanceEthicsError("Ancient record detected")
