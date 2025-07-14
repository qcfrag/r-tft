from datetime import datetime
import pathlib

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

class ResonanceEthicsError(Exception):
    pass

def check_domain(domain: str):
    d = (domain or "").lower()
    
    # Explicit whitelist check
    if d in ALLOWED_DOMAINS:
        return
        
    # Full-string and substring block
    for bad in FORBIDDEN:
        if bad and bad in d:
            raise ResonanceEthicsError(
                f"REL-1.0 violation: '{bad}' matched in domain '{domain}'"
            )
            
    # If not explicitly allowed, treat as disallowed
    raise ResonanceEthicsError(
        f"REL-1.0 violation: domain '{domain}' is not in allowed list"
    )

def check_metadata(meta: dict[str, Any] | None):
    if not meta:
        return
        
    for key, val in meta.items():
        sval = str(val).lower()
        for bad in FORBIDDEN:
            if bad in sval:
                raise ResonanceEthicsError(
                    f"Metadata violation: '{bad}' detected in '{key}' value"
                )
