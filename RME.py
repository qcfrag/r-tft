from ethics import check_domain, ResonanceEthicsError

# ----------------------------------------------------------
class RME_Record:
    """A single memory entry."""
    def __init__(self, vector, domain: str, meta=None):
        check_domain(domain)          # ← fails immediately if disallowed
        self.vector = vector
        self.domain = domain
        self.meta   = meta or {}


class ResonanceMemoryEngine:
    """Simple in-memory store; adapt to your existing API."""
    def __init__(self):
        self.memory = []

    # ---------- add / store --------------------------------
    def add_record(self, record: RME_Record):
        check_domain(record.domain)   # redundant but extra safe
        self.memory.append(record)

    # ---------- search example -----------------------------
    def similarity_search(self, query_vec, top_k=5):
        scored = []
        for rec in self.memory:
            check_domain(rec.domain)  # guard every read
            score = cosine_sim(query_vec, rec.vector)
            scored.append((score, rec))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:top_k]


# Utility: fast cosine similarity
def cosine_sim(a, b, eps=1e-8):
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)
