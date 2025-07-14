from __future__ import annotations
from typing import List, Any
import numpy as np

# Central ethics guard (must exist in same package)
from ethics import check_domain, ResonanceEthicsError

# ---------------------------------------------------------------------
# Utility: fast cosine similarity (works on Python lists or numpy arrays)
# ---------------------------------------------------------------------

def cosine_sim(a: List[float] | np.ndarray, b: List[float] | np.ndarray, eps: float = 1e-8) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + eps)
    return float(np.dot(a_arr, b_arr) / denom)

# ---------------------------------------------------------------------
# Dataclass‑style record
# ---------------------------------------------------------------------

class RME_Record:
    """Single resonance memory entry."""

    __slots__ = ("vector", "domain", "meta")

    def __init__(self, vector: List[float] | np.ndarray, domain: str, meta: dict[str, Any] | None = None):
        check_domain(domain)  # ← Enforce REL‑1.0 at creation time
        self.vector = np.asarray(vector, dtype=float)
        self.domain = domain.lower()
        self.meta = meta or {}

    def similarity(self, other_vec: List[float] | np.ndarray) -> float:
        """Cosine similarity between this record and an arbitrary vector."""
        return cosine_sim(self.vector, other_vec)

# ---------------------------------------------------------------------
# Resonance Memory Engine container
# ---------------------------------------------------------------------

class ResonanceMemoryEngine:
    """A lightweight, in‑RAM store for resonance vectors."""

    def __init__(self):
        self._memory: list[RME_Record] = []

    # ------------- Add / remove ------------------------------------------------

    def add_record(self, record: RME_Record) -> None:
        """Add a new memory record (REL‑1.0 guard inside RME_Record)."""
        check_domain(record.domain)  # redundant but extra‑safe
        self._memory.append(record)

    def clear(self) -> None:
        self._memory.clear()

    # ------------- Query -------------------------------------------------------

    def similarity_search(self, query_vec: List[float] | np.ndarray, top_k: int = 5) -> list[tuple[float, RME_Record]]:
        """Return top‑k most similar records to *query_vec*.

        Raises ResonanceEthicsError if any record domain is forbidden.
        """
        scored: list[tuple[float, RME_Record]] = []
        for rec in self._memory:
            check_domain(rec.domain)  # Guard each access
            score = rec.similarity(query_vec)
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[: top_k]

    # ------------- Batch utility ----------------------------------------------

    def batch_similarity(self, query_vecs: list[list[float] | np.ndarray]) -> list[list[float]]:
        """Compute cosine similarity matrix between many queries and all records."""
        out = []
        for q in query_vecs:
            out.append([r.similarity(q) for r in self._memory])
        return out

# ---------------------------------------------------------------------
# Minimal demo (executes only if run as script)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    try:
        # Create engine and records
        rme = ResonanceMemoryEngine()
        rme.add_record(RME_Record([0.2, 0.8], domain="scientific", meta={"tag": "demo"}))

        # Attempt forbidden domain (should raise)
        rme.add_record(RME_Record([1, 0], domain="military"))
    except ResonanceEthicsError as e:
        print("[REL‑1.0 BLOCKED]", e)
