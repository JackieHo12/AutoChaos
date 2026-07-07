import numpy as np
from typing import List, Tuple, Dict

def normalize_metrics(metrics: np.ndarray, norm_factors: np.ndarray) -> np.ndarray:
    return (metrics - norm_factors) / (norm_factors + metrics + 1e-10)

def compute_lyapunov_exponent(
    map_data: np.ndarray,
    Vin: np.ndarray,
    Vc: float,
    n_iterations: int = 1000
) -> float:
    from scipy.interpolate import interp1d
    Vout = map_data[:, 1]
    map_func = interp1d(Vin, Vout, kind='cubic', fill_value='extrapolate')
    x = 0.5
    lyap_sum = 0.0
    valid_count = 0
    for i in range(n_iterations):
        dx = 0.001
        x_plus = np.clip(x + dx, Vin.min(), Vin.max())
        x_minus = np.clip(x - dx, Vin.min(), Vin.max())
        derivative = (map_func(x_plus) - map_func(x_minus)) / (2 * dx)
        if abs(derivative) > 1e-10:
            lyap_sum += np.log(abs(derivative))
            valid_count += 1
        x = map_func(x)
        x = np.clip(x, Vin.min(), Vin.max())


    if valid_count > 0:
        return lyap_sum / valid_count
    else:
        return -999.0

def compute_bifurcation_diagram(
    map_data: np.ndarray,
    Vin: np.ndarray,
    Vc_range: np.ndarray,
    n_transient: int = 100,
    n_sample: int = 50
) -> List[Tuple[float, List[float]]]:

    from scipy.interpolate import interp1d
    bifurcation_data = []
    for Vc in Vc_range:
        Vout = map_data[:, 1]
        map_func = interp1d(Vin, Vout, kind='cubic', fill_value='extrapolate')
        x = 0.5
        for _ in range(n_transient):
            x = map_func(x)
            x = np.clip(x, Vin.min(), Vin.max())
        values = []
        for _ in range(n_sample):
            x = map_func(x)
            x = np.clip(x, Vin.min(), Vin.max())
            values.append(x)
        bifurcation_data.append((Vc, values))
    return bifurcation_data

def compute_chaotic_ratio(lyapunov_vs_vc: List[Tuple[float, float]]) -> float:

    if not lyapunov_vs_vc:
        return 0.0
    chaotic_count = sum(1 for _, le in lyapunov_vs_vc if le > 0)
    return chaotic_count / len(lyapunov_vs_vc)

def compute_bifurcation_density(bifurcation_data: List) -> int:

    bifurcations = 0
    prev_count = 0
    for Vc, values in bifurcation_data:
        unique_vals = len(set(np.round(values, decimals=3)))
        if abs(unique_vals - prev_count) > 0:
            bifurcations += 1
        prev_count = unique_vals
    return bifurcations
