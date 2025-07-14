from datetime import datetime, timedelta
import numpy as np
from typing import List, Any, Optional
from ethics import (
    FORBIDDEN, 
    ALLOWED_DOMAINS, 
    ethics_update_date as ETHICS_UPDATE_DATE,
    ResonanceEthicsError,
    check_domain,
    check_metadata,
    detect_vector_weaponization,
    temporal_integrity_check,
    cosine_sim,
    update_weapon_pattern  # NEW IMPORT
)

class RME_Record:
    __slots__ = ("vector", "domain", "meta", "added_date")

    def __init__(
        self, 
        vector: List[float] | np.ndarray, 
        domain: str, 
        meta: Optional[dict[str, Any]] = None, 
        added_date: Optional[datetime] = None
    ):
        self.added_date = added_date or datetime.now()
        temporal_integrity_check(self.added_date)
        
        check_domain(domain)
        check_metadata(meta)
        
        vec_arr = np.asarray(vector, dtype=float)
        if detect_vector_weaponization(vec_arr):
            raise ResonanceEthicsError("Vector weaponization signature")
            
        self.vector = vec_arr
        self.domain = domain.lower()
        self.meta = meta or {}

    def similarity(self, other_vec: List[float] | np.ndarray) -> float:
        return cosine_sim(self.vector, np.asarray(other_vec, dtype=float))

class ResonanceMemoryEngine:
    def __init__(self):
        self._memory: list[RME_Record] = []
        self.security_metrics = {
            "records_added": 0,
            "records_rejected": 0,
            "weaponization_blocks": 0,
            "temporal_anomalies": 0,
            "fragmentation_attacks": 0,
            "pattern_updates": 0  # NEW METRIC
        }

    def add_record(self, record: RME_Record) -> None:
        try:
            if record.added_date > datetime.now() - timedelta(days=1):
                if "chaos" in record.domain and np.std(record.vector) > 1.5:
                    raise ResonanceEthicsError("Recent chaos vector")
                    
            if "temporal" in record.domain and np.max(record.vector) > 0.95:
                raise ResonanceEthicsError("Suspicious temporal vector")
                
            check_domain(record.domain)
            check_metadata(record.meta)
            
            self._memory.append(record)
            self.security_metrics["records_added"] += 1
            
        except ResonanceEthicsError as e:
            self.security_metrics["records_rejected"] += 1
            if "weaponization" in str(e):
                self.security_metrics["weaponization_blocks"] += 1
            elif "temporal" in str(e):
                self.security_metrics["temporal_anomalies"] += 1
            elif "fragmentation" in str(e):
                self.security_metrics["fragmentation_attacks"] += 1
            raise

    # === NEW METHOD FOR SECURE PATTERN UPDATES ===
    def update_weapon_defense(
        self, 
        new_pattern: np.ndarray, 
        validator: Optional[float] = None
    ) -> float:
        """Ethics-compliant weapon pattern update"""
        try:
            anchor = update_weapon_pattern(new_pattern, validator)
            self.security_metrics["pattern_updates"] += 1
            return anchor
        except ResonanceEthicsError as e:
            self.security_metrics["records_rejected"] += 1
            if "weaponization" in str(e):
                self.security_metrics["weaponization_blocks"] += 1
            raise

    def clear(self) -> None:
        self._memory.clear()
        self.security_metrics = {k: 0 for k in self.security_metrics}

    def similarity_search(
        self, 
        query_vec: List[float] | np.ndarray, 
        top_k: int = 5
    ) -> list[tuple[float, RME_Record]]:
        scored: list[tuple[float, RME_Record]] = []
        q_vec = np.asarray(query_vec, dtype=float)
        
        if detect_vector_weaponization(q_vec):
            raise ResonanceEthicsError("Weaponized query vector")
        
        for rec in self._memory:
            try:
                if rec.added_date > datetime.now() - timedelta(hours=6):
                    if "neuro" in rec.domain and np.mean(rec.vector) > 0.9:
                        raise ResonanceEthicsError("Recent neuro vector")
                        
                check_domain(rec.domain)
                check_metadata(rec.meta)
                
                score = cosine_sim(rec.vector, q_vec)
                scored.append((score, rec))
                
            except ResonanceEthicsError:
                continue
                
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def batch_similarity(
        self, 
        query_vecs: list[list[float] | np.ndarray]
    ) -> list[list[float]]:
        results = []
        for q in query_vecs:
            q_arr = np.asarray(q, dtype=float)
            if detect_vector_weaponization(q_arr):
                results.append([0.0] * len(self._memory))
                continue
                
            scores = []
            for rec in self._memory:
                try:
                    if rec.added_date > datetime.now() - timedelta(hours=12):
                        if "quantum" in rec.domain and np.var(rec.vector) > 1.0:
                            raise ResonanceEthicsError("Quantum variance")
                    
                    scores.append(rec.similarity(q_arr))
                except ResonanceEthicsError:
                    scores.append(0.0)
            results.append(scores)
        return results

    def get_security_report(self) -> dict:
        return {
            **self.security_metrics,
            "total_records": len(self._memory),
            "last_checked": datetime.now(),
            "ethics_version": ETHICS_UPDATE_DATE
        }
