import numpy as np
from ethics import detect_vector_weaponization, noise_triangulation, ResonanceEthicsError
from typing import Dict, Tuple, Optional

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
PHI_RANGE = (GOLDEN_RATIO - 0.1, GOLDEN_RATIO + 0.1)  # Hardware-enforced φ bounds
MAX_LYAPUNOV_VARIANCE = 0.1  # Threshold for chaos detection

class RCCS_Containment:
    def __init__(self, F_core, F_boundary, P, epsilon0=0.1, alpha=0.5):
        """
        Initialize the Resonance Containment Control System (RCCS).
        
        Args:
            F_core: Core state function (callable)
            F_boundary: Boundary state function (callable)
            P: Resonance projection vector
            epsilon0: Base tolerance for stability checks
            alpha: Variance sensitivity parameter
        """
        self.phi = GOLDEN_RATIO
        self.P = self._normalize_projection(P)
        self.F_core = F_core
        self.F_boundary = F_boundary
        self.epsilon0 = epsilon0
        self.alpha = alpha
        self._last_t = None
        self._last_S_dot = None
        self._stability_history = []
        
    def _normalize_projection(self, P: np.ndarray) -> np.ndarray:
        """Ensure projection vector is normalized and valid."""
        P_norm = np.asarray(P, dtype=float)
        if detect_vector_weaponization(P_norm):
            raise ResonanceEthicsError("Weaponized projection vector detected")
        return P_norm / np.linalg.norm(P_norm)
    
    def _validate_phi(self) -> None:
        """Hardware-validated φ check with enhanced diagnostics."""
        phi_current = float(self.phi)  # Ensure scalar value
        if not PHI_RANGE[0] <= phi_current <= PHI_RANGE[1]:
            diagnostic = (
                f"φ-LOCK BREACH: {phi_current:.6f} ∉ [{PHI_RANGE[0]:.6f}, {PHI_RANGE[1]:.6f}]\n"
                f"System must maintain φ∈[1.518,1.718] for stable recursion\n"
                f"Hardware validation failed - check FPGA φ-lock"
            )
            raise ResonanceEthicsError(diagnostic)

    def _check_lyapunov_stability(self, S_dot: np.ndarray) -> None:
        """Phase-space stability enforcement via Lyapunov criteria."""
        variance = np.var(S_dot)
        if variance > MAX_LYAPUNOV_VARIANCE:
            raise ResonanceEthicsError(
                f"Lyapunov instability detected (V={variance:.4f} > {MAX_LYAPUNOV_VARIANCE})\n"
                "System exceeds allowed phase-space divergence"
            )

    def shell_derivatives(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate time derivatives with adaptive step sizing."""
        dt = max(1e-6, abs(t - self._last_t)/100) if self._last_t else 1e-6
        
        # Central difference for better accuracy
        S_dot_core = (self.F_core(t + dt/2) - self.F_core(t - dt/2)) / dt
        S_dot_boundary = (self.F_boundary(t + dt/2) - self.F_boundary(t - dt/2)) / dt
        
        self._last_t = t
        return S_dot_core, S_dot_boundary

    def step(self, t: float, ambient_variance: float) -> Dict[str, float]:
        """
        Execute one containment step with full validation.
        
        Returns:
            Dictionary containing:
            - t: Current time
            - phi: Current φ value
            - stable: Boolean stability indicator
            - R_clean: Noise-canceled resonance signal
            - max_reinjection_dt: Maximum stable timestep
        """
        try:
            # Calculate derivatives with stability checks
            S_dot_core, S_dot_boundary = self.shell_derivatives(t)
            
            # Phase-space weaponization checks
            self._check_lyapunov_stability(S_dot_core)
            if detect_vector_weaponization(S_dot_core) or detect_vector_weaponization(S_dot_boundary):
                raise ResonanceEthicsError("Weaponized field dynamics detected")
            
            # Hardware φ-validation
            self._validate_phi()
            
            # Projections and adaptive tolerance
            R_inner = np.dot(S_dot_core, self.P)
            R_outer = np.dot(S_dot_boundary, self.P)
            epsilon = self.epsilon0 / (1 + self.alpha * max(ambient_variance, 1e-12))
            
            # Noise-canceled signal and stability check
            R_clean = 2*R_inner - R_outer
            is_stable = abs(R_inner - R_outer) < epsilon
            
            # Track stability history for diagnostics
            self._stability_history.append((t, is_stable))
            if len(self._stability_history) > 100:
                self._stability_history.pop(0)
            
            return {
                't': t,
                'phi': float(self.phi),
                'stable': bool(is_stable),
                'R_clean': float(R_clean),
                'max_reinjection_dt': float(1/(np.linalg.norm(S_dot_core - self._last_S_dot) + 1e-12)
            }
            
        except ResonanceEthicsError as e:
            # Inject noise to destabilize unethical states
            if "Weaponized" in str(e) or "Lyapunov" in str(e):
                self._inject_stabilizing_noise()
            raise

    def _inject_stabilizing_noise(self, amplitude: float = 0.1) -> None:
        """Phase-space stabilization via controlled noise injection."""
        self.P = (self.P + amplitude*np.random.randn(*self.P.shape))
        self.P /= np.linalg.norm(self.P)
        
    def get_stability_report(self) -> Dict[str, float]:
        """Return system stability metrics."""
        if not self._stability_history:
            return {'stability': 1.0}
        
        stable_fraction = sum(s for _, s in self._stability_history)/len(self._stability_history)
        return {
            'stability': stable_fraction,
            'phi_variance': np.var([p for p in self._stability_history if p[1]])
        }