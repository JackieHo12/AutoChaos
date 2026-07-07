import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass

class MatlabEngineConfig:
    matlab_cmd: str = "matlab"
    chaotic_dir: str = "eval_engines/matlab"
    chaotic_func: str = "chaotic"
    timeout_s: int = 600
    default_csv_path: Optional[str] = None

class MatlabEngine:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = MatlabEngineConfig(
            matlab_cmd=config.get("matlab_cmd", "matlab"),
            chaotic_dir=config.get("chaotic_dir", "eval_engines/matlab"),
            chaotic_func=config.get("chaotic_func", "chaotic"),
            timeout_s=int(config.get("timeout_s", 600)),
            default_csv_path=config.get("default_csv_path", config.get("csv_path", "data/test_result.csv")),
        )
    @staticmethod
    def _as_float_or_none(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            if isinstance(x, str) and x.lower() == "nan":
                return None
            val = float(x)
            if val != val:
                return None
            return val
        except Exception:
            return None
    def evaluate(self, params: Dict[str, Any], csv_path: Optional[str] = None) -> Dict[str, Any]:
        if csv_path is None:
            csv_path = self.cfg.default_csv_path
        if not csv_path:
            raise ValueError(
                "MatlabEngine.evaluate() requires csv_path, but none was provided and "
                "no default_csv_path/csv_path was set in engine config."
            )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        metrics_path = os.path.join(
            tempfile.gettempdir(),
            f"autochaos_metrics_{os.getpid()}_{next(tempfile._get_candidate_names())}.json",
        )
        batch = (
            f"addpath('{self.cfg.chaotic_dir}'); "
            f"{self.cfg.chaotic_func}('{csv_path}','{metrics_path}');"
        )
        cmd = [self.cfg.matlab_cmd, "-batch", batch]
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.cfg.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"MATLAB timed out after {self.cfg.timeout_s}s") from e
        if completed.returncode != 0:
            raise RuntimeError(
                "MATLAB returned non-zero exit code.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{completed.stdout}"
            )
        if not os.path.exists(metrics_path):
            raise RuntimeError(
                "MATLAB ran but metrics JSON was not created.\n"
                f"Expected: {metrics_path}\n"
                f"MATLAB output:\n{completed.stdout}"
            )
        with open(metrics_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out = dict(raw)
        out["mle"] = self._as_float_or_none(raw.get("mle"))
        out["ale"] = self._as_float_or_none(raw.get("ale"))
        out["le_cr"] = self._as_float_or_none(raw.get("le_cr"))
        out["vc_mle"] = self._as_float_or_none(raw.get("vc_mle"))
        try:
            os.remove(metrics_path)
        except OSError:
            pass
        return out
