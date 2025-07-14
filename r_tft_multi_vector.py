
import numpy as np

def r_tft_multi_vector(theta_dot, P_list, window=100):
    """
    Multi-vector Real-Time Fractional Tracking (R-TFT)

    Args:
        theta_dot (np.ndarray): Angular velocity vector (shape: [T, D])
        P_list (list of np.ndarray): List of resonance vectors (each shape: [D])
        window (int): Ring buffer window length

    Returns:
        R_clean_list: List of cleaned resonance projections
        R_inner_list: List of raw resonance projections
    """
    R_clean_list = []
    R_inner_list = []

    for P in P_list:
        P_unit = P / np.linalg.norm(P)
        R_inner = np.dot(theta_dot, P_unit.T)
        buffer = np.zeros(window)
        R_clean = []
        for i, val in enumerate(R_inner):
            buffer[i % window] = val
            R_avg = np.mean(buffer)
            R_clean.append(2 * val - R_avg)
        R_clean_list.append(np.array(R_clean))
        R_inner_list.append(R_inner)

    return R_clean_list, R_inner_list
