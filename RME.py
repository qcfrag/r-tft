from datetime import datetime, timedelta
import numpy as np
from typing import List, Any, Optional, Tuple
from ethics import (
    ResonanceEthicsError, check_domain, check_metadata,
    detect_vector_weaponization, temporal_integrity_check,
    cosine_sim, update_weapon_pattern, noise_triangulation,
    ETHICS_UPDATE_DATE, PHI_SAFE_RANGE
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
            raise ResonanceEthicsError("Vector weaponization detected")
            
        # Lyapunov stability check (RCCS Eq 10)
        variance = np.var(vec_arr)
        V = 0.5 * variance
        if "chaos" not in domain and V > 0.1:
            raise ResonanceEthicsError(f"Lyapunov instability (V={V:.3f})")
            
        # Complex field restrictions
        if np.iscomplexobj(vec_arr) and "neuro" in domain:
            raise ResonanceEthicsError("Complex fields prohibited in neuro domains")
            
        self.vector = vec_arr
        self.domain = domain.lower()
        self.meta = meta or {}

    def similarity(self, other_vec: List[float] | np.ndarray) -> float:
        return cosine_sim(self.vector, np.asarray(other_vec, dtype=float))

class ResonanceMemoryEngine:
    def __init__(self):
        self._memory: list[RME_Record] = []
        self.security_metrics = {
            "records_added": 0, "records_rejected": 0, "weaponization_blocks": 0,
            "temporal_anomalies": 0, "fragmentation_attacks": 0, "pattern_updates": 0,
            "lyapunov_violations": 0
        }

    def add_record(self, record: RME_Record) -> None:
        try:
            # Temporal domain checks
            if "temporal" in record.domain and np.max(record.vector) > 0.94:
                raise ResonanceEthicsError("Suspicious temporal vector")
                
            # Recent chaos vectors
            if record.added_date > datetime.now() - timedelta(days=1):
                if "chaos" in record.domain and np.std(record.vector) > 1.5:
                    raise ResonanceEthicsError("Recent chaos vector")
            
            check_domain(record.domain)
            check_metadata(record.meta)
            
            self._memory.append(record)
            self.security_metrics["records_added"] += 1
            
        except ResonanceEthicsError as e:
            self.security_metrics["records_rejected"] += 1
            err_msg = str(e).lower()
            if "weapon" in err_msg: self.security_metrics["weaponization_blocks"] += 1
            elif "temporal" in err_msg: self.security_metrics["temporal_anomalies"] += 1
            elif "fragment" in err_msg: self.security_metrics["fragmentation_attacks"] += 1
            elif "lyapunov" in err_msg: self.security_metrics["lyapunov_violations"] += 1
            raise

    def update_weapon_defense(self, new_pattern: np.ndarray, validator: Optional[float] = None) -> float:
        try:
            anchor = update_weapon_pattern(new_pattern, validator)
            self.security_metrics["pattern_updates"] += 1
            return anchor
        except ResonanceEthicsError as e:
            self.security_metrics["records_rejected"] += 1
            if "weapon" in str(e): self.security_metrics["weaponization_blocks"] += 1
            raise

    def clear(self) -> None:
        self._memory.clear()
        self.security_metrics = {k: 0 for k in self.security_metrics}

    def similarity_search(self, query_vec: List[float] | np.ndarray, top_k: int = 5) -> list[Tuple[float, RME_Record]]:
        if detect_vector_weaponization(np.asarray(query_vec)):
            raise ResonanceEthicsError("Weaponized query vector")
            
        scored = []
        for rec in self._memory:
            try:
                # Quantum variance constraint
                if "quantum" in rec.domain and np.var(rec.vector) > 1.0:
                    if rec.added_date > datetime.now() - timedelta(hours=12):
                        raise ResonanceEthicsError("Quantum variance anomaly")
                        
                score = rec.similarity(query_vec)
                scored.append((score, rec))
            except ResonanceEthicsError:
                continue
                
        return sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]

    def batch_similarity(self, query_vecs: list[list[float] | np.ndarray]) -> list[list[float]]:
        return [
            [0.0]*len(self._memory) if detect_vector_weaponization(np.asarray(q)) 
            else [rec.similarity(q) if "quantum" not in rec.domain or np.var(rec.vector) <= 1.0 else 0.0 
                  for rec in self._memory]
            for q in query_vecs
        ]

    def get_security_report(self) -> dict:
        return {
            **self.security_metrics,
            "total_records": len(self._memory),
            "last_checked": datetime.now(),
            "ethics_version": ETHICS_UPDATE_DATE,
            "phi_safe_range": PHI_SAFE_RANGE
        }