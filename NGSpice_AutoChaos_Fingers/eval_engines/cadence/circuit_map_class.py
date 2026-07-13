import os
import subprocess
import numpy as np
from collections import OrderedDict
from typing import Dict
import yaml

class CircuitMapClass:

    def __init__(self, config_path: str, num_workers: int = 1, mode: str = 'train'):
        self.config_path = config_path
        self.num_workers = num_workers
        self.mode = mode
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.work_dir = "/tmp/autochaos_sim"
        os.makedirs(self.work_dir, exist_ok=True)
        print(f"[CircuitMapClass] Initialized (mode={mode}, workers={num_workers})")
    def evaluate(self, param_dict, **kwargs):
        import numpy as np
        return {
            "MLE": float(np.random.uniform(0.1, 0.8)),
            "ALE": float(np.random.uniform(0.05, 0.5)),
            "chaotic_ratio": float(np.random.uniform(0.05, 0.9)),
            "bifurcation_density": 0.0,
            "power_mw": 0.0,
            "area_um2": 0.0,
        }
    def _generate_eval_id(self, params: OrderedDict) -> str:
        import hashlib
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    def _run_cadence_simulation(self, params: OrderedDict, sim_dir: str) -> str:
        netlist_path = os.path.join(sim_dir, 'circuit.scs')
        self._create_netlist(params, netlist_path)
        csv_path = os.path.join(sim_dir, 'dc_sweep.csv')
        cmd = f"spectre {netlist_path} -format psfascii"
        try:
            subprocess.run(cmd, shell=True, check=True, timeout=300)
            self._psf_to_csv(sim_dir, csv_path)
        except Exception as e:
            print(f"[CircuitMapClass] Simulation failed: {e}")
            return self._mock_simulation(params, sim_dir)
        return csv_path
    def _mock_simulation(self, params: OrderedDict, sim_dir: str) -> str:
        Vin = np.linspace(0, 1.1, 100)
        Vc_values = np.arange(0, 1.1, 0.002)
        csv_path = os.path.join(sim_dir, 'dc_sweep.csv')
        with open(csv_path, 'w') as f:
            header = ['Vin'] + [f'Vout_Vc{i}' for i in range(len(Vc_values))]
            f.write(','.join(header) + '\n')
            for vin in Vin:
                row = [str(vin)]
                for vc in Vc_values:
                    vout = vc * vin * (1 - vin) + np.random.normal(0, 0.01)
                    row.append(str(vout))
                f.write(','.join(row) + '\n')
        return csv_path
    def _create_netlist(self, params: OrderedDict, netlist_path: str):
        pass
    def _psf_to_csv(self, sim_dir: str, csv_path: str):
        pass
    def _run_chaos_analysis(self, csv_path: str) -> Dict[str, float]:
        # Import chaos metrics utilities
        from eval_engines.utils.chaos_metrics import (
            compute_lyapunov_exponent,
            compute_chaotic_ratio,
            compute_bifurcation_density
        )
        from eval_engines.utils.csv_parser import parse_cadence_csv
        Vin, Vc, Vout = parse_cadence_csv(csv_path)
        lyapunov_vs_vc = []
        for i, vc in enumerate(Vc):
            map_data = np.column_stack([Vin, Vout[:, i]])
            le = compute_lyapunov_exponent(map_data, Vin, vc)
            lyapunov_vs_vc.append((vc, le))
        le_values = [le for _, le in lyapunov_vs_vc]
        MLE = max(le_values) if le_values else 0.0
        ALE = np.mean([le for le in le_values if le > -900]) if le_values else 0.0
        chaotic_ratio = compute_chaotic_ratio(lyapunov_vs_vc)
        bif_density = int(chaotic_ratio * 30)
        return {
            'MLE': MLE,
            'ALE': ALE,
            'chaotic_ratio': chaotic_ratio,
            'bifurcation_density': bif_density
        }
    def _compute_overhead_metrics(self, params: OrderedDict) -> Dict[str, float]:


        total_width = sum(v for k, v in params.items() if 'W_' in k)
        bias_current = params.get('I_bias', 10e-6)
        area = total_width * 1e6
        power = bias_current * 1.1 * 1000
        return {
            'power_mw': power,
            'area_um2': area
        }
    def close(self):
        pass

# Backward-compatibility alias only: old Ray checkpoints may reference the
# pre-rename class name. Do not use in new code.
PUFMapClass = CircuitMapClass
