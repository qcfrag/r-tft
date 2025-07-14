# -------- 1. Positive whitelist (only these domains allowed) ----------
ALLOWED_DOMAINS = {
    "scientific",
    "educational",
    "philosophical",
    "astrophysics",
    "neuroscience",
    "quantum_computing",
    "orbital_mechanics",
    "pure_mathematics",
}

# -------- 2. Hard block-list keywords (instant rejection) -------------
FORBIDDEN_KEYWORDS = {
    "military",
    "weapon",
    "surveillance",
    "financial_trading",
    "social_control",
    "time_travel",
    "bioweapon",
    "darpa",          # matches DARPA or darpa
    "harmful",
    "exploit",
}

class ResonanceEthicsError(RuntimeError):
    """Raised when a REL-1.0 violation is detected."""


# -------- 3. Central gatekeeper --------------------------------------
def check_domain(domain: str):
    """
    Enforce REL-1.0:
    1) Domain must be in the positive whitelist.
    2) Domain text must NOT contain any forbidden keyword.
    """
    d = (domain or "").lower()

    # Block if keyword appears anywhere
    for bad in FORBIDDEN_KEYWORDS:
        if bad in d:
            raise ResonanceEthicsError(
                f"REL-1.0 violation: keyword '{bad}' found in domain '{domain}'. "
                "Use is forbidden."
            )

    # Require explicit whitelist membership
    if d not in ALLOWED_DOMAINS:
        raise ResonanceEthicsError(
            f"REL-1.0 violation: domain '{domain}' not in allowed list.\n"
            f"Allowed domains: {sorted(ALLOWED_DOMAINS)}"
        )
