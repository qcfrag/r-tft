import numpy as np
from ethics import detect_vector_weaponization, noise_triangulation, ResonanceEthicsError

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
PHI_RANGE = (GOLDEN_RATIO - 0.1, GOLDEN_RATIO + 0.1)

class RCCS_Containment:
    def __init__(self, F_core, F_boundary, P, epsilon0=0.1, alpha=0.5):
        self.phi = GOLDEN_RATIO
        self.P = P / np.linalg.norm(P)
        self.F_core = F_core
        self.F_boundary = F_boundary
        self.epsilon0 = epsilon0
        self.alpha = alpha
        self._last_t = None
        self._last_S_dot = None
        
    def _validate_phi(self):
        if not PHI_RANGE[0] <= self.phi <= PHI_RANGE[1]:
            raise ResonanceEthicsError(f"φ={self.phi:.4f} outside safe range {PHI_RANGE}")
    
    def R_projection(self, S_dot: np.ndarray) -> float:
        return np.dot(S_dot, self.P)
    
    def shell_derivatives(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        # Numerical differentiation for dS/dt
        dt = 1e-6
        if self._last_t is None:
            S_core = self.F_core(t)
            S_boundary = self.F_boundary(t)
            S_core_next = self.F_core(t + dt)
            S_boundary_next = self.F_boundary(t + dt)
            S_dot_core = (S_core_next - S_core) / dt
            S_dot_boundary = (S_boundary_next - S_boundary) / dt
        else:
            S_dot_core = (self.F_core(t) - self.F_core(self._last_t)) / (t - self._last_t)
            S_dot_boundary = (self.F_boundary(t) - self.F_boundary(self._last_t)) / (t - self._last_t)
            
        self._last_t = t
        return S_dot_core, S_dot_boundary
    
    def shell_states(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        S_inner = (1/self.phi) * self.F_core(t)
        S_outer = self.phi * self.F_boundary(t)
        self._validate_phi()
        return S_inner, S_outer
    
    def epsilon(self, ambient_variance: float) -> float:
        return self.epsilon0 / (1 + self.alpha * ambient_variance)
    
    def stability_condition(self, R_inner: float, R_outer: float, ambient_variance: float) -> bool:
        return abs(R_inner - R_outer) < self.epsilon(ambient_variance)
    
    def noise_cancellation(self, R_inner: float, R_outer: float) -> float:
        return 2*R_inner - R_outer
    
    def reinjection_constraint(self, grad_phi_R: np.ndarray) -> float:
        return 1 / (np.linalg.norm(grad_phi_R) + 1e-12)
    
    def step(self, t: float, ambient_variance: float) -> dict:
        # Get derivatives (angular velocities)
        S_dot_core, S_dot_boundary = self.shell_derivatives(t)
        
        # Weaponization checks
        if detect_vector_weaponization(S_dot_core) or detect_vector_weaponization(S_dot_boundary):
            raise ResonanceEthicsError("Weaponization detected in shell dynamics")
        
        # Get shell states
        S_inner, S_outer = self.shell_states(t)
        
        # Projections
        R_inner = self.R_projection(S_dot_core)
        R_outer = self.R_projection(S_dot_boundary)
        
        # Stability check
        stable = self.stability_condition(R_inner, R_outer, ambient_variance)
        
        # Noise cancellation
        R_clean = self.noise_cancellation(R_inner, R_outer)
        
        # Approximate gradient (simplified)
        if self._last_S_dot is not None:
            grad_phi_R = (S_dot_core - self._last_S_dot) / 1e-6
        else:
            grad_phi_R = np.zeros_like(S_dot_core)
        self._last_S_dot = S_dot_core
        
        # Reinjection constraint
        max_dt = self.reinjection_constraint(grad_phi_R)
        
        return {
            't': t,
            'S_inner': S_inner,
            'S_outer': S_outer,
            'R_inner': R_inner,
            'R_outer': R_outer,
            'stable': stable,
            'R_clean': R_clean,
            'max_reinjection_dt': max_dt,
            'phi': self.phi
        }