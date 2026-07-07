import os
import re
import subprocess
from multiprocessing import Semaphore

MATLAB_MAX_CONCURRENT = 2
MATLAB_SEMAPHORE = Semaphore(MATLAB_MAX_CONCURRENT)

class MATLABChaosAnalyzer:

    def __init__(self, matlab_bin="matlab", chaotic_script_dir="eval_engines/matlab"):
        self.matlab_bin = matlab_bin
        self.chaotic_script_dir = chaotic_script_dir
        self.chaotic_m_path = os.path.join(self.chaotic_script_dir, "chaotic.m")
        print(f"[MATLABChaosAnalyzer] Initialized. chaotic.m: {self.chaotic_m_path}")
    def analyze_csv(self, csv_path, output_dir=None):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        matlab_expr = (
            f"addpath('{self.chaotic_script_dir}'); "
            f"chaotic('{csv_path}');"
        )
        print(f"[MATLABChaosAnalyzer] Waiting for slot (limit={MATLAB_MAX_CONCURRENT})...")
        with MATLAB_SEMAPHORE:
            print(f"[MATLABChaosAnalyzer] Running MATLAB on: {csv_path}")
            try:
                result = subprocess.run(
                    [
                        self.matlab_bin,
                        "-batch",
                        matlab_expr,
                    ],
                    timeout=180,
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                print("[MATLABChaosAnalyzer] MATLAB timeout")
                return {"MLE": 0.0, "ALE": 0.0, "chaotic_ratio": 0.0}
            except subprocess.CalledProcessError as e:
                print("[MATLABChaosAnalyzer] MATLAB crashed")
                stdout = e.stdout or ""
                stderr = e.stderr or ""
                if stdout:
                    print(stdout[-2000:])
                if stderr:
                    print(stderr[-2000:])
                return {"MLE": 0.0, "ALE": 0.0, "chaotic_ratio": 0.0}
        stdout = result.stdout or ""
        cr_match = re.search(r"Chaotic Ratio \(LE_cr\):\s*([0-9eE+\-.]+)", stdout)
        ale_match = re.search(r"Average Lyapunov Exponent \(ALE\):\s*([0-9eE+\-.]+)", stdout)
        mle_match = re.search(r"Maximum Lyapunov Exponent \(MLE\):\s*([0-9eE+\-.]+)", stdout)
        metrics = {
            "MLE": float(mle_match.group(1)) if mle_match else 0.0,
            "ALE": float(ale_match.group(1)) if ale_match else 0.0,
            "chaotic_ratio": float(cr_match.group(1)) if cr_match else 0.0,
        }
        if metrics["MLE"] == 0.0 and metrics["ALE"] == 0.0 and metrics["chaotic_ratio"] == 0.0:
            print("[MATLABChaosAnalyzer] WARNING: Could not parse MATLAB metrics from stdout")
            if stdout:
                print(stdout[-2000:])
        print(
            f"[MATLABChaosAnalyzer] "
            f"MLE={metrics['MLE']:.4f}, "
            f"ALE={metrics['ALE']:.4f}, "
            f"CR={metrics['chaotic_ratio']:.4f}"
        )
        return metrics

