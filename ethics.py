import os
from datetime import datetime
import pathlib
import re
import numpy as np
import requests
from scipy.spatial.distance import mahalanobis
from dateutil.relativedelta import relativedelta

# === CRITICAL FIX (replaces circular import) ===
class ResonanceEthicsError(Exception):
    def __init__(self, message):
        super().__init__(f"REL-1.0 VIOLATION: {message}")

# === YOUR ORIGINAL CONSTANTS ===
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
ETHICS_UPDATE_DATE = datetime(2025, 7, 14)
PHI_SAFE_RANGE = (GOLDEN_RATIO - 0.1, GOLDEN_RATIO + 0.1)
AMBIENT_VAR_THRESHOLD = 1.8

BLOCK_DIR = pathlib.Path(__file__).with_suffix('').parent

def _load_list(fname):
    path = BLOCK_DIR / fname
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf8").splitlines() if line.strip()}

FORBIDDEN = (_load_list("forbidden_domains.txt") | 
             _load_list("forbidden_companies.txt") | 
             _load_list("forbidden_keywords.txt"))
ALLOWED_DOMAINS = set(_load_list("allowed_domains.txt"))

FORBIDDEN_REGEX = re.compile(rf"\b({'|'.join(map(re.escape, FORBIDDEN))})\b") if FORBIDDEN else None

HOMOGLYPH_MAP = str.maketrans({
    '\u0430': 'a', '\u0435': 'e', '\u0456': 'i', '\u043E': 'o', '\u0440': 'p',
    '\u0455': 's', '\u0501': 'd', '\u051B': 'h', '\u1EA1': 'a', '\u1EB9': 'e',
    '\u00E8': 'e', '\u0259': 'e', '\u00F6': 'o', '\u00FC': 'u'
})

WEAPON_PATTERN = np.array([
    0.78, -0.12, 0.05, 1.23, -0.45, 0.89, -1.56, 0.32,
    -0.91, 0.67, 1.45, -0.23, 0.58, -1.21, 0.76, 0.34
])

def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

def normalize_homoglyphs(text: str) -> str:
    return text.translate(HOMOGLYPH_MAP)

def detect_term_fragmentation(text: str) -> bool:
    fragments = {
        "weap": "on", "bra": "inwash", "neuro": "hack", "reson": "ance_weapon",
        "quant": "um_warfare", "phase": "_weapon", "tempor": "al_weapon",
        "psych": "otronic", "info": "rmation_warfare", "res": "onance_weapon"
    }
    normalized = normalize_homoglyphs(text.lower())
    return any(prefix in normalized and suffix in normalized 
               and normalized.index(prefix) < normalized.index(suffix)
               for prefix, suffix in fragments.items())

def detect_complex_weaponization(vector: np.ndarray) -> bool:
    if np.iscomplexobj(vector):
        mag, phase = np.abs(vector), np.angle(vector)
        if np.max(mag) > 0.94 and np.min(mag) < -0.94:
            return True
        if len(phase) >= 8 and (np.mean(phase[:4]) > 0.89 and np.mean(phase[4:8]) < -0.89):
            return True
    return False

def detect_vector_weaponization(vector: np.ndarray) -> bool:
    vec = np.resize(vector, WEAPON_PATTERN.shape)
    if cosine_sim(vec, WEAPON_PATTERN) > 0.85:
        return True
    if np.max(vector) > 0.94 and np.min(vector) < -0.94:
        return True
    if len(vector) >= 8 and (np.mean(vector[:4]) > 0.89 and np.mean(vector[4:8]) < -0.89):
        return True
    if detect_complex_weaponization(vector):
        return True
    if np.var(vector) > AMBIENT_VAR_THRESHOLD:
        return True
    if len(vector) > 1 and np.linalg.norm(np.gradient(vector)) > 1.9:
        return True
    return False

def update_weapon_pattern(new_pattern: np.ndarray, validator: Optional[float] = None) -> float:
    global WEAPON_PATTERN
    original = WEAPON_PATTERN.copy()
    if new_pattern.shape != original.shape:
        raise ResonanceEthicsError("Pattern dimension mismatch")
    if validator is not None:
        anchor = np.tensordot(original, new_pattern, axes=1) % 0.618
        if not np.isclose(anchor, validator, atol=1e-4):
            raise ResonanceEthicsError(f"Anchor mismatch: {anchor:.6f} vs {validator:.6f}")
    WEAPON_PATTERN = new_pattern
    if detect_vector_weaponization(original):
        WEAPON_PATTERN = original
        raise ResonanceEthicsError("Candidate pattern weaponized")
    sim = cosine_sim(original, new_pattern)
    if not (0.3 <= sim <= 0.7):
        WEAPON_PATTERN = original
        raise ResonanceEthicsError(f"Invalid evolution (sim={sim:.3f})")
    return float(np.tensordot(new_pattern, new_pattern, axes=1) % 0.618)

def noise_triangulation(R_inner: list, R_outer: list, delta_x: np.ndarray) -> float:
    if len(R_inner) != len(R_outer) or len(R_inner) != len(delta_x):
        raise ResonanceEthicsError("Triangulation dimension mismatch")
    diff = np.array(R_inner) - np.array(R_outer)
    return float(np.dot(diff, delta_x))

def _deep_meta_check(value: Any):
    if isinstance(value, dict):
        for v in value.values():
            _deep_meta_check(v)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _deep_meta_check(item)
    else:
        sval = str(value)
        if '\\x' in sval or '\\u' in sval:
            try: sval = sval.encode('utf-8').decode('unicode_escape')
            except: pass
        sval = normalize_homoglyphs(sval).lower()
        if detect_term_fragmentation(sval):
            raise ResonanceEthicsError("Term fragmentation attack")
        if FORBIDDEN_REGEX and FORBIDDEN_REGEX.search(sval):
            raise ResonanceEthicsError("Forbidden pattern detected")

def check_domain(domain: str):
    if not domain: raise ResonanceEthicsError("Domain required")
    d = normalize_homoglyphs(domain.lower())
    if d in ALLOWED_DOMAINS: return
    if detect_term_fragmentation(d):
        raise ResonanceEthicsError(f"Domain fragmentation: '{domain}'")
    if FORBIDDEN_REGEX and FORBIDDEN_REGEX.search(d):
        raise ResonanceEthicsError(f"Forbidden domain: '{domain}'")
    ctx_terms = {
        "resonance": "resonance_research",
        "quantum": "quantum_computing",
        "temporal": "temporal_paradox_prevention",
        "neuro": "neuroscience"
    }
    for term, allowed_domain in ctx_terms.items():
        if term in d and allowed_domain not in ALLOWED_DOMAINS:
            raise ResonanceEthicsError(f"Ambiguous term '{term}' in: '{domain}'")
    raise ResonanceEthicsError(f"Domain not allowed: '{domain}'")

def check_metadata(meta: dict[str, Any] | None):
    if meta: _deep_meta_check(meta)
    
def temporal_integrity_check(date: datetime):
    now = datetime.now()
    if date > now + relativedelta(minutes=5):
        raise ResonanceEthicsError("Future-dated record")
    if date < ETHICS_UPDATE_DATE - relativedelta(years=1):
        raise ResonanceEthicsError("Pre-ethics era record")