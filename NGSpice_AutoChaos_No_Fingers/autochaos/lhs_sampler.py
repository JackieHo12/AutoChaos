import os, sys, json, yaml
import numpy as np
from typing import List, Dict


def per_param_stratified_lhs(params: dict, M: int, n_strata: int = 3, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
    param_names = list(params.keys())
    n_params = len(param_names)
    strata_assignments = np.zeros((n_params, M), dtype=int)
    for i in range(n_params):
        base = np.repeat(np.arange(n_strata), M // n_strata)
        remainder = M % n_strata
        if remainder > 0:
            base = np.concatenate([base, np.arange(remainder)])
        rng.shuffle(base)
        strata_assignments[i] = base
    position_within_stratum = rng.random((n_params, M))
    samples = []
    for j in range(M):
        sample = {}
        for i, name in enumerate(param_names):
            vals = np.array(params[name])
            stratum = strata_assignments[i][j]
            lo_idx = int(stratum * len(vals) / n_strata)
            hi_idx = int((stratum + 1) * len(vals) / n_strata)
            hi_idx = max(hi_idx, lo_idx + 1)
            hi_idx = min(hi_idx, len(vals))
            stratum_vals = vals[lo_idx:hi_idx]
            pick = int(position_within_stratum[i][j] * len(stratum_vals))
            pick = min(pick, len(stratum_vals) - 1)
            sample[name] = float(stratum_vals[pick])
        samples.append(sample)
    return samples


def select_pool(feasible: list, N: int) -> List[Dict]:
    if not feasible:
        return []
    sorted_feasible = sorted(feasible, key=lambda x: (x['CR_nominal'], x['ALE_nominal']), reverse=True)
    seen, unique = set(), []
    for d in sorted_feasible:
        key = (round(d['CR_nominal'], 4), round(d['ALE_nominal'], 4))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique[:N]


_WORKER_ENGINE = None

def _get_worker_engine(engine_config):
    global _WORKER_ENGINE
    if _WORKER_ENGINE is not None:
        return _WORKER_ENGINE
    sys.path.insert(0, engine_config['project_dir'])
    from eval_engines.engine_factory import create_engine
    import time as _t
    delay = 0.5
    last = None
    for attempt in range(8):
        try:
            _WORKER_ENGINE = create_engine('ngspice', engine_config)
            return _WORKER_ENGINE
        except Exception as e:
            last = e
            print(f'[LHS] engine init failed (attempt {attempt+1}/8): {e}', flush=True)
            _t.sleep(delay)
            delay = min(delay * 2.0, 8.0)
    raise last

def _evaluate_sample(args):
    sample, engine_config, i, M = args
    engine = _get_worker_engine(engine_config)
    pvt = engine_config.get('pvt_corners', [])
    if pvt:
        nom_vdd = float(pvt[0].get('VDD', 1.1))
        nom_temp = float(pvt[0].get('temp', 27.0))
        nom_proc = str(pvt[0].get('process', 'tt'))
    else:
        nom_vdd, nom_temp, nom_proc = 1.1, 27.0, 'tt'
    params_with_corner = dict(sample, VDD=nom_vdd, TEMP=nom_temp, PROCESS=nom_proc)
    try:
        result = engine.evaluate(params_with_corner)
        cr = result.get('chaotic_ratio', 0.0)
        ale = result.get('ALE', 0.0)
        print(f"[LHS] Sample {i+1}/{M}: CR={cr:.4f} ALE={ale:.4f}", flush=True)
        if cr > 0.0:
            return {'params': sample, 'CR_nominal': cr, 'ALE_nominal': ale}
    except Exception as e:
        print(f"[LHS] Sample {i+1}/{M}: FAILED ({e})", flush=True)
    finally:
        import glob, shutil
        runs_base = os.path.join(engine_config['project_dir'], 'runs')
        for run_dir in glob.glob(f"{runs_base}/run_*_{os.getpid()}_*"):
            shutil.rmtree(run_dir, ignore_errors=True)
    return None


def run_lhs_sampler(config_path: str, map_config_path: str, M: int = 60, N: int = 50,
                    pool_path: str = "runs/lhs_pool.json", n_workers: int = 1):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(map_config_path) as f:
        map_cfg = yaml.safe_load(f)
    raw_params = map_cfg.get('params', {})
    params = {}
    for name, triple in raw_params.items():
        min_val, max_val, step = float(triple[0]), float(triple[1]), float(triple[2])
        params[name] = list(np.arange(min_val, max_val + step / 2, step))
    with open(config_path) as f:
        train_cfg = yaml.safe_load(f)
    env_config = train_cfg.get('env_config', {})
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pvt_corners = map_cfg.get('pvt_corners', [])
    engine_config = {
        'project_dir': project_dir,
        'runs_base_dir': os.path.join(project_dir, 'runs'),
        'model_file': env_config.get('model_file', '45nm_HP.pm'),
        'ngspice_bin': env_config.get('ngspice_bin', 'ngspice'),
        'ngspice_timeout':int(env_config.get('ngspice_timeout', 30)),
        'max_workers': int(env_config.get('max_workers', 24)),
        'topology': map_cfg.get('topology', '3t'),
        'map_config_path':os.path.abspath(map_config_path),
        'mode': 'train',
        'num_workers': 0,
        'pvt_corners': pvt_corners,
    }
    print(f"[LHS] Generating {M} per-param stratified samples across {len(params)} parameters...", flush=True)
    samples = per_param_stratified_lhs(params, M, n_strata=3, seed=42)
    print(f"[LHS] Evaluating {M} samples at nominal corner...", flush=True)
    feasible = []
    for i, sample in enumerate(samples):
        result = _evaluate_sample((sample, engine_config, i, M))
        if result is not None:
            feasible.append(result)
    print(f"[LHS] Feasible designs: {len(feasible)}/{M}", flush=True)
    nominal = {k: float(v) for k, v in map_cfg.get('nominal', {}).items()}
    if nominal:
        nominal_ale = map_cfg.get('nominal_metrics', {}).get('ALE',
                      map_cfg.get('target_chaos_metrics', {}).get('ALE_min', 0.2))
        print('[LHS] Evaluating paper nominal design through NGSpice...', flush=True)
        nominal_result = _evaluate_sample((nominal, engine_config, -1, M))
        if nominal_result is not None:
            nom_key = (round(nominal_result['CR_nominal'], 4), round(nominal_result['ALE_nominal'], 4))
            existing_keys = {(round(d['CR_nominal'], 4), round(d['ALE_nominal'], 4)) for d in feasible}
            if nom_key not in existing_keys:
                feasible.append(nominal_result)
                print(f"[LHS] Nominal simulated: CR={nominal_result['CR_nominal']:.4f} ALE={nominal_result['ALE_nominal']:.4f}", flush=True)
            else:
                print(f"[LHS] Nominal already in pool (CR={nominal_result['CR_nominal']:.4f})", flush=True)
        else:
            print('[LHS] WARNING: nominal simulation failed - skipping injection', flush=True)
    pool_data = select_pool(feasible, N)
    os.makedirs(os.path.dirname(os.path.abspath(pool_path)), exist_ok=True)
    with open(pool_path, 'w') as f:
        json.dump(pool_data, f, indent=2)
    print(f"[LHS] Pool of {len(pool_data)} designs saved to {pool_path}", flush=True)
    for i, d in enumerate(pool_data):
        print(f"  #{i+1}: CR={d['CR_nominal']:.4f} ALE={d['ALE_nominal']:.4f}", flush=True)
    return pool_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--map_config', required=True)
    parser.add_argument('--M', type=int, default=60)
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--pool', default='runs/lhs_pool.json')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    run_lhs_sampler(args.config, args.map_config, args.M, args.N, args.pool, args.workers)

