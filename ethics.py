# ethics.py  – central REL-1.0 gatekeeper
ALLOWED_DOMAINS = {
    "astrophysics",
    "neuroscience",
    "quantum_computing",
    "orbital_mechanics",
    "pure_mathematics",
    "scientific",
    "educational",
    "philosophical",
}

class ResonanceEthicsError(RuntimeError):
    pass

def check_domain(domain: str):
    if domain not in ALLOWED_DOMAINS:
        raise ResonanceEthicsError(
            f"REL-1.0 violation: domain '{domain}' is forbidden.\n"
            f"Allowed domains: {sorted(ALLOWED_DOMAINS)}"
        )
