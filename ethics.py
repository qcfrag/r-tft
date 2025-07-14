# ethics.py  –  central REL-1.0 gatekeeper
ALLOWED_DOMAINS = {
    "scientific",
    "educational",
    "philosophical"
}

class ResonanceEthicsError(RuntimeError):
    """Raised when a REL-1.0 violation is detected."""

def check_domain(domain: str):
    """
    Ensure the requested domain is REL-1.0 compliant.
    """
    if domain not in ALLOWED_DOMAINS:
        raise ResonanceEthicsError(
            f"REL-1.0 violation: domain '{domain}' is forbidden.\n"
            f"Allowed domains: {sorted(ALLOWED_DOMAINS)}"
        )
