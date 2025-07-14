from datetime import datetime

# Date when forbidden lists were last updated
ethics_update_date = datetime(2025, 7, 14)  # ← Set this to your latest revision date

import pathlib

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

ALLOWED_DOMAINS = {
    "scientific", "educational", "philosophical",
    "astrophysics", "neuroscience", "quantum_computing",
    "orbital_mechanics", "cmb_analysis", "dynamical_systems",
    "resonance_research", "chaos_theory", "mathematical_physics",
    "consciousness_studies", "cosmology", "neural_oscillations",
    "phase_detection", "hamiltonian_systems"
}

class ResonanceEthicsError(RuntimeError):
    pass

def check_domain(domain: str):
    d = (domain or "").lower()

    # 1) explicit whitelist check (positive allow)
    if d in ALLOWED_DOMAINS:
        return

    # 2) full-string and substring block
    for bad in FORBIDDEN:
        if bad and bad in d:
            raise ResonanceEthicsError(
                f"REL-1.0 violation: '{bad}' matched in domain '{domain}'. "
                "Operation aborted."
            )

    # 3) if not explicitly allowed, treat as disallowed
    raise ResonanceEthicsError(
        f"REL-1.0 violation: domain '{domain}' is not in allowed list."
    )
