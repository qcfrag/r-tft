# ethics.py  —  central REL-1.0 guard
ALLOWED_DOMAINS = {
    "scientific",
    "educational",
    "philosophical",
}

class ResonanceEthicsError(RuntimeError):
    """Raised when a REL-1.0 violation is detected."""
    pass

def check_domain(domain: str):
    """
    Fail fast if someone tries to use a forbidden domain.
    """
    if domain not in ALLOWED_DOMAINS:
        raise ResonanceEthicsError(
            f"REL-1.0 violation: domain '{domain}' is forbidden.\n"
            f"Allowed domains: {sorted(ALLOWED_DOMAINS)}"
        )
