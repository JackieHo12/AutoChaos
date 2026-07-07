from __future__ import annotations
from typing import Dict, Any
import numpy as np


class MockEngine:

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def evaluate(self, params: Dict[str, Any] | None = None, **kwargs) -> Dict[str, float]:
        rng = np.random.default_rng(int(self.config.get("seed", 0)))
        # A simple stable set of outputs
        return {
            "MLE": float(rng.normal(0.5, 0.01)),
            "ALE": float(rng.normal(0.2, 0.01)),
            "chaotic_ratio": float(np.clip(rng.normal(0.7, 0.02), 0.0, 1.0)),
            "power_mw": 1.0,
            "area_um2": 1.0,
        }
