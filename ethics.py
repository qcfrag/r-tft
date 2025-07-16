import os
from datetime import datetime
import pathlib
import re
import numpy as np
import requests
from scipy.spatial.distance import mahalanobis
from dateutil.relativedelta import relativedelta
from ethics import ResonanceEthicsError

# GitHub issue logging
def log_violation(error_msg: str):
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # 🔒 Use environment variable
    if not GITHUB_TOKEN:
        raise EnvironmentError("GitHub token not found. Please set the GITHUB_TOKEN environment variable.")
    REPO = "qcfrag/Real-Time-Fractional-Tracking-R-TFT"
    URL = f"https://api.github.com/repos/{REPO}/issues"

    payload = {
        "title": f"REL-1.0 VIOLATION: {datetime.now().isoformat()}",
        "body": f"```\n{error_msg}\n```\n\nAuto-flagged by ethics.py.",
        "labels": ["critical", "violation"]
    }

    try:
        response = requests.post(URL, json=payload, headers={
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        })
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Failed to log violation to GitHub: {e}")
        # Handle the error or log it locally as needed

# Example usage in detect_vector_weaponization():
if detect_vector_weaponization(vector):
    error = "Weaponized vector detected. Containment collapse triggered."
    log_violation(error)
    raise ResonanceEthicsError(error)

# Ethical constants
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
ETHICS_UPDATE_DATE = datetime(2025, 7, 14)
PHI_SAFE_RANGE = (GOLDEN_RATIO - 0.1, GOLDEN_RATIO + 0.1)  # ±0.1 around φ
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

# Detect weaponized patterns in vectors
def detect_vector_weaponization(vector: np.ndarray) -> bool:
    # Check against weapon pattern
    vec = np.resize(vector, WEAPON_PATTERN.shape)
    if cosine_sim(vec, WEAPON_PATTERN) > 0.85:
        return True
        
    # Coercion signature
    if np.max(vector) > 0.94 and np.min(vector) < -0.94:
        return True
        
    # Temporal signature
    if len(vector) >= 8 and (np.mean(vector[:4]) > 0.89 and np.mean(vector[4:8]) < -0.89):
        return True
        
    # Complex field weaponization
    if detect_complex_weaponization(vector):
        return True
        
    # Variance threshold
    if np.var(vector) > AMBIENT_VAR_THRESHOLD:
        return True
        
    # Gradient magnitude safeguard
    if len(vector) > 1 and np.linalg.norm(np.gradient(vector)) > 1.9:
        return True
        
    return False

# Update weapon pattern
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
