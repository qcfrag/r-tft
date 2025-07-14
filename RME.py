# RME.py
from ethics import check_domain, ResonanceEthicsError

class RME_Record:
    def __init__(self, vector, domain: str, meta=None):
        check_domain(domain)          # Enforce REL-1.0 here
        self.vector = vector
        self.domain = domain
        self.meta   = meta or {}
