from datetime import datetime
import numpy as np
from typing import List, Any, Optional
from ethics import (
    FORBIDDEN, 
    ALLOWED_DOMAINS, 
    ethics_update_date as ETHICS_UPDATE_DATE,
    ResonanceEthicsError,
    check_domain,
    check_metadata
)

def cosine_sim(a: List[float] | np.ndarray, b: List[float] | np.ndarray, eps: float = 1e-8) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + eps
    return float(np.dot(a_arr, b_arr) / denom

class RME_Record:
    __slots__ = ("vector", "domain", "meta", "added_date")

    def __init__(
        self, 
        vector: List[float] | np.ndarray, 
        domain: str, 
        meta: Optional[dict[str, Any]] = None, 
        added_date: Optional[datetime] = None
    ):
        # Validate domain and metadata before creating record
        check_domain(domain)
        check_metadata(meta)
        
        self.vector = np.asarray(vector, dtype=float)
        self.domain = domain.lower()
        self.meta = meta or {}
        self.added_date = added_date or datetime.now()

    def similarity(self, other_vec: List[float] | np.ndarray) -> float:
        return cosine_sim(self.vector, other_vec)

class ResonanceMemoryEngine:
    def __init__(self):
        self._memory: list[RME_Record] = []

    def add_record(self, record: RME_Record) -> None:
        """Adds a record after validating ethics compliance"""
        # Revalidate at insertion time
        check_domain(record.domain)
        check_metadata(record.meta)
        self._memory.append(record)

    def clear(self) -> None:
        self._memory.clear()

    def similarity_search(
        self, 
        query_vec: List[float] | np.ndarray, 
        top_k: int = 5
    ) -> list[tuple[float, RME_Record]]:
        """Ethics-compliant similarity search"""
        scored: list[tuple[float, RME_Record]] = []
        
        for rec in self._memory:
            # Revalidate at query time with current ethics rules
            try:
                check_domain(rec.domain)
                check_metadata(rec.meta)
            except ResonanceEthicsError as e:
                # Skip unethical records rather than failing entire query
                continue
                
            score = cosine_sim(rec.vector, query_vec)
            scored.append((score, rec))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def batch_similarity(
        self, 
        query_vecs: list[list[float] | np.ndarray]
    ) -> list[list[float]]:
        """Batch similarity with ethics compliance"""
        results = []
        for q in query_vecs:
            scores = []
            for rec in self._memory:
                try:
                    check_domain(rec.domain)
                    check_metadata(rec.meta)
                    scores.append(rec.similarity(q))
                except ResonanceEthicsError:
                    # Treat unethical records as zero similarity
                    scores.append(0.0)
            results.append(scores)
        return results
