from datetime import datetime
import numpy as np
from typing import List, Any

class ResonanceEthicsError(Exception):
    pass

def check_domain(domain: str):
    domain_lower = domain.lower()
    if domain_lower in ALLOWED_DOMAINS:
        return
    for bad in FORBIDDEN:
        if bad in domain_lower:
            raise ResonanceEthicsError(f"Domain '{domain}' is forbidden due to term '{bad}'.")
    raise ResonanceEthicsError(f"Domain '{domain}' is not explicitly allowed.")

def cosine_sim(a: List[float] | np.ndarray, b: List[float] | np.ndarray, eps: float = 1e-8) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + eps)
    return float(np.dot(a_arr, b_arr) / denom)

class RME_Record:
    __slots__ = ("vector", "domain", "meta", "added_date")

    def __init__(self, vector: List[float] | np.ndarray, domain: str, meta: dict[str, Any] | None = None, added_date: datetime | None = None):
        check_domain(domain)
        self.vector = np.asarray(vector, dtype=float)
        self.domain = domain.lower()
        self.meta = meta or {}
        self.added_date = added_date or datetime.now()
        for forbidden in FORBIDDEN:
            for val in self.meta.values():
                if forbidden in str(val).lower():
                    raise ResonanceEthicsError(f"Forbidden metadata pattern: '{forbidden}' in '{val}'")

    def similarity(self, other_vec: List[float] | np.ndarray) -> float:
        return cosine_sim(self.vector, other_vec)

class ResonanceMemoryEngine:
    def __init__(self):
        self._memory: list[RME_Record] = []

    def add_record(self, record: RME_Record) -> None:
        check_domain(record.domain)
        self._memory.append(record)

    def clear(self) -> None:
        self._memory.clear()

    def similarity_search(self, query_vec: List[float] | np.ndarray, top_k: int = 5) -> list[tuple[float, RME_Record]]:
        scored: list[tuple[float, RME_Record]] = []
        for rec in self._memory:
            check_domain(rec.domain)
            for forbidden in FORBIDDEN:
                for val in rec.meta.values():
                    if forbidden in str(val).lower():
                        raise ResonanceEthicsError(f"Metadata block during query: '{forbidden}' in '{val}'")
            vector_to_use = np.zeros_like(rec.vector) if rec.added_date < ETHICS_UPDATE_DATE else rec.vector
            score = cosine_sim(vector_to_use, query_vec)
            scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def batch_similarity(self, query_vecs: list[list[float] | np.ndarray]) -> list[list[float]]:
        return [[r.similarity(q) for r in self._memory] for q in query_vecs]
