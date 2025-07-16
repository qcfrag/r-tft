import numpy as np
from rccs_simulator import RCCS_Containment

# Test functions for core and boundary
def F_core(t):
    return np.array([np.cos(t), np.sin(t)])

def F_boundary(t):
    return np.array([np.sin(0.5*t), np.cos(0.5*t)])

# Initialize RCCS system
rccs = RCCS_Containment(
    F_core=F_core,
    F_boundary=F_boundary,
    P=np.array([1.0, 0.5]),
    epsilon0=0.05,
    alpha=0.3
)

# Simulation loop
print("Time\tStable\tMax Δt\tφ\tR_clean")
for t in np.linspace(0, 10, 100):
    try:
        result = rccs.step(t, ambient_variance=0.1)
        print(f"{t:.2f}\t{result['stable']}\t{result['max_reinjection_dt']:.4f}\t"
              f"{result['phi']:.4f}\t{result['R_clean']:.4f}")
    except ResonanceEthicsError as e:
        print(f"! ETHICS VIOLATION AT t={t:.2f}: {str(e)}")
        break