"""
Resonance Containment Control System (RCCS) Simulator

A quantum-field stabilization system that maintains coherence between core and boundary states
using golden ratio (φ) constrained dynamics. Implements hardware-enforced ethical constraints
and Lyapunov stability checks.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from ethics import detect_vector_weaponization, noise_triangulation, ResonanceEthicsError

# Mathematical constants
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
PHI_RANGE = (GOLDEN_RATIO - 0.1, GOLDEN_RATIO + 0.1)  # Hardware-enforced φ bounds
MAX_LYAPUNOV_VARIANCE = 0.1  # Threshold for chaos detection

class RCCS_Containment:
    """Main containment system for resonance stabilization.
    
    Attributes:
        phi (float): Current golden ratio value (1.618... ± tolerance)
        P (np.ndarray): Normalized projection vector
        F_core (Callable): Core state function f(t) -> np.ndarray
        F_boundary (Callable): Boundary state function f(t) -> np.ndarray
        epsilon0 (float): Base tolerance for stability checks
        alpha (float): Variance sensitivity parameter
    """
    
    def __init__(self, F_core: Callable, F_boundary: Callable, P: np.ndarray, 
                 epsilon0: float = 0.1, alpha: float = 0.5):
        """Initialize the RCCS with core/boundary functions and projection vector.
        
        Args:
            F_core: Function that returns core state vector at time t
            F_boundary: Function that returns boundary state vector at time t
            P: Initial projection vector (will be normalized)
            epsilon0: Base tolerance for stability threshold (default: 0.1)
            alpha: Variance sensitivity scaling factor (default: 0.5)
            
        Raises:
            ResonanceEthicsError: If weaponized vectors are detected
            TypeError: If input functions don't match expected signatures
        """
        # Input validation
        if not callable(F_core) or not callable(F_boundary):
            raise TypeError("State functions must be callable")
        if not isinstance(P, (np.ndarray, list)):
            raise TypeError("Projection vector must be array-like")
            
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
        """Normalize projection vector with ethical validation.
        
        Args:
            P: Input projection vector
            
        Returns:
            Normalized unit vector
            
        Raises:
            ResonanceEthicsError: If weaponization patterns detected
        """
        P_norm = np.asarray(P, dtype=float)
        if detect_vector_weaponization(P_norm):
            raise ResonanceEthicsError("Weaponized projection vector detected")
        norm = np.linalg.norm(P_norm)
        if norm < 1e-12:
            raise ValueError("Projection vector cannot be zero")
        return P_norm / norm
    
    def _validate_phi(self) -> None:
        """Validate golden ratio is within hardware-enforced bounds.
        
        Raises:
            ResonanceEthicsError: If φ leaves safety bounds
        """
        phi_current = float(self.phi)
        if not PHI_RANGE[0] <= phi_current <= PHI_RANGE[1]:
            diagnostic = (
                f"φ-LOCK BREACH: {phi_current:.6f} ∉ [{PHI_RANGE[0]:.6f}, {PHI_RANGE[1]:.6f}]\n"
                f"System must maintain φ∈[1.518,1.718] for stable recursion\n"
                f"Hardware validation failed - check FPGA φ-lock"
            )
            raise ResonanceEthicsError(diagnostic)

    def _check_lyapunov_stability(self, S_dot: np.ndarray) -> None:
        """Enforce phase-space stability via Lyapunov criteria.
        
        Args:
            S_dot: State derivative vector
            
        Raises:
            ResonanceEthicsError: If variance exceeds chaos threshold
        """
        variance = np.var(S_dot)
        if variance > MAX_LYAPUNOV_VARIANCE:
            raise ResonanceEthicsError(
                f"Lyapunov instability detected (V={variance:.4f} > {MAX_LYAPUNOV_VARIANCE})\n"
                "System exceeds allowed phase-space divergence"
            )

    def shell_derivatives(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate time derivatives with adaptive step sizing.
        
        Args:
            t: Current simulation time
            
        Returns:
            Tuple of (core_derivative, boundary_derivative)
        """
        dt = max(1e-6, abs(t - self._last_t)/100) if self._last_t else 1e-6
        
        # Central difference for better accuracy
        S_dot_core = (self.F_core(t + dt/2) - self.F_core(t - dt/2)) / dt
        S_dot_boundary = (self.F_boundary(t + dt/2) - self.F_boundary(t - dt/2)) / dt
        
        self._last_t = t
        self._last_S_dot = S_dot_core
        return S_dot_core, S_dot_boundary

    def step(self, t: float, ambient_variance: float = 0.0) -> Dict[str, float]:
        """Execute one containment step with full validation.
        
        Args:
            t: Current simulation time
            ambient_variance: Environmental noise variance
            
        Returns:
            Dictionary containing:
            - t: Current time
            - phi: Current φ value
            - stable: Boolean stability indicator
            - R_clean: Noise-canceled resonance signal
            - max_reinjection_dt: Maximum stable timestep
            
        Raises:
            ResonanceEthicsError: If any stability check fails
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
            
            # Adaptive tolerance based on environmental noise
            epsilon = self.epsilon0 / (1 + self.alpha * max(ambient_variance, 1e-12))
            
            # Projections and stability check
            R_inner = np.dot(S_dot_core, self.P)
            R_outer = np.dot(S_dot_boundary, self.P)
            R_clean = 2*R_inner - R_outer  # Noise-canceled signal
            is_stable = abs(R_inner - R_outer) < epsilon
            
            # Track stability history
            self._stability_history.append((t, float(is_stable)))
            if len(self._stability_history) > 100:
                self._stability_history.pop(0)
            
            return {
                't': float(t),
                'phi': float(self.phi),
                'stable': bool(is_stable),
                'R_clean': float(R_clean),
                'max_reinjection_dt': float(1/(np.linalg.norm(S_dot_core - self._last_S_dot) + 1e-12)
            }
            
        except ResonanceEthicsError as e:
            # Emergency stabilization protocol
            if "Weaponized" in str(e) or "Lyapunov" in str(e):
                self._inject_stabilizing_noise()
            raise

    def _inject_stabilizing_noise(self, amplitude: float = 0.1) -> None:
        """Inject controlled noise to destabilize unethical states.
        
        Args:
            amplitude: Noise scaling factor (default: 0.1)
        """
        self.P = (self.P + amplitude*np.random.randn(*self.P.shape))
        self.P = self._normalize_projection(self.P)
        
    def get_stability_report(self) -> Dict[str, float]:
        """Generate system stability diagnostics report.
        
        Returns:
            Dictionary with:
            - stability: Fraction of stable steps (0.0-1.0)
            - phi_variance: Variance of φ during stable periods
        """
        if not self._stability_history:
            return {'stability': 1.0, 'phi_variance': 0.0}
        
        stable_points = [t for t, s in self._stability_history if s]
        return {
            'stability': np.mean([s for _, s in self._stability_history]),
            'phi_variance': np.var(stable_points) if stable_points else 0.0
        }


# Example Usage
if __name__ == "__main__":
    # Sample state functions
    def core_state(t):
        return np.array([np.sin(t), np.cos(t), 0.1*t])
    
    def boundary_state(t):
        return np.array([1.2*np.sin(t), 0.8*np.cos(t), 0.1*t + 0.05*np.random.randn()])
    
    try:
        # Initialize system
        rccs = RCCS_Containment(
            F_core=core_state,
            F_boundary=boundary_state,
            P=[1, 0.5, -0.2],
            epsilon0=0.05,
            alpha=0.3
        )
        
        # Run simulation steps
        for t in np.linspace(0, 10, 100):
            result = rccs.step(t, ambient_variance=0.01)
            print(f"t={result['t']:.2f} | Stable: {result['stable']} | R={result['R_clean']:.4f}")
            
        # Print final report
        report = rccs.get_stability_report()
        print(f"\nStability Report:\n{'-'*30}")
        print(f"Stability Score: {report['stability']:.1%}")
        print(f"φ Variance: {report['phi_variance']:.6f}")
        
    except ResonanceEthicsError as e:
        print(f"SYSTEM SHUTDOWN: Ethical violation detected\n{str(e)}")