import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

TRUNCATE = 1000
SEQ_LEN = 3000
N_ITER = SEQ_LEN + TRUNCATE
INIT_STATE = 0.5
DELTA_SCALE = 0.001


def load_csv(csv_path: str):
    data = np.loadtxt(csv_path, delimiter=",")
    vout_cols = np.arange(1, data.shape[1], 2)
    VOUT = data[:, vout_cols]
    VIN = data[:, 0]
    return VIN, VOUT


def _interp_all(VIN, VOUT, x):
    n_vin, n_vc = VOUT.shape
    idx = np.searchsorted(VIN, x, side="right") - 1
    idx = np.clip(idx, 0, n_vin - 2)
    x0 = VIN[idx]
    x1 = VIN[idx + 1]
    col = np.arange(n_vc)
    y0 = VOUT[idx, col]
    y1 = VOUT[idx + 1, col]
    dx = x1 - x0
    w = np.where(dx > 0, (x - x0) / dx, 0.0)
    return y0 + w * (y1 - y0)


def compute_lyapunov(VIN, VOUT):
    n_vin, n_vc = VOUT.shape
    delta = (VIN[1] - VIN[0]) * DELTA_SCALE
    vin_max = VIN.max()
    x = np.full(n_vc, INIT_STATE)
    diff_sum = np.zeros(n_vc)
    for it in range(N_ITER):
        x_del = x + delta
        x_del[x_del > vin_max] -= 2.0 * delta
        f_x = _interp_all(VIN, VOUT, x)
        f_xd = _interp_all(VIN, VOUT, x_del)
        ratio = np.abs((f_xd - f_x) / (x_del - x))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(ratio > 0, np.log(ratio), -5.0)
        log_ratio = np.where(np.isneginf(log_ratio) | np.isnan(log_ratio), -5.0, log_ratio)
        if it >= TRUNCATE:
            diff_sum += log_ratio
        x = f_x
    return diff_sum / SEQ_LEN


def analyze_csv(csv_path: str) -> Dict[str, float]:
    VIN, VOUT = load_csv(csv_path)
    LE = compute_lyapunov(VIN, VOUT)
    positive = LE > 0
    CR = float(positive.sum()) / len(LE)
    ALE = float(LE[positive].mean()) if positive.any() else 0.0
    MLE = float(LE.max())
    return {"MLE": MLE, "ALE": ALE, "chaotic_ratio": CR,
            "bifurcation_density": 0.0, "power_mw": 0.0, "area_um2": 0.0}


def analyze_csv_batch(csv_paths: list, max_workers: int = None) -> list:
    if max_workers is None:
        max_workers = min(len(csv_paths), os.cpu_count() or 4)
    results = [None] * len(csv_paths)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_csv, p): i for i, p in enumerate(csv_paths)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"[ChaosAnalyzer] Error on {csv_paths[i]}: {e}")
                results[i] = {"MLE": 0.0, "ALE": 0.0, "chaotic_ratio": 0.0,
                              "bifurcation_density": 0.0, "power_mw": 0.0, "area_um2": 0.0}
    return results


class PythonChaosAnalyzer:

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, os.cpu_count() or 4)
        print(f"[PythonChaosAnalyzer] Initialized. max_workers={self.max_workers}")

    def analyze(self, csv_path: str) -> Dict[str, float]:
        return analyze_csv(csv_path)

    def analyze_batch(self, csv_paths: list) -> list:
        return analyze_csv_batch(csv_paths, max_workers=self.max_workers)
