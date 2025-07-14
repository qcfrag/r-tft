"""
R-TFT: Real-Time Fractional Tracking
Author: Éric Lanctôt-Rivest
License: REL-1.0
"""
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class RTFTracker:
    def __init__(self, P, window=100):
        self.P = np.array(P)
        self.norm_P = np.linalg.norm(P)
        self.buffer = deque(maxlen=window)

    def update(self, theta_dot):
        theta_dot = np.array(theta_dot)
        R_inner = np.dot(theta_dot, self.P) / self.norm_P
        self.buffer.append(R_inner)
        R_outer = np.mean(self.buffer) if self.buffer else 0
        R_clean = 2 * R_inner - R_outer
        return R_clean, R_inner

if __name__ == '__main__':
    # Example: simulate resonance signal
    P = [1.618, -1.0]
    tracker = RTFTracker(P)
    R_vals = []

    for t in range(300):
        theta_dot = [np.cos(0.01 * t), np.sin(0.01 * t)]
        R_clean, _ = tracker.update(theta_dot)
        R_vals.append(R_clean)

    plt.plot(R_vals)
    plt.title("R-TFT Clean Resonance Signal")
    plt.xlabel("Timestep")
    plt.ylabel("R_clean")
    plt.grid(True)
    plt.show()