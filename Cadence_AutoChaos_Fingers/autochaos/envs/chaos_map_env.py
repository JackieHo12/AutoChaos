
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import json
from collections import OrderedDict
from typing import Dict, Tuple, Optional
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_engines.engine_factory import create_engine
from eval_engines.cadence.cadence_engine import SpectreSimulationError
from eval_engines.utils.chaos_metrics import normalize_metrics


class ChaosMapEnv(gym.Env):

    metadata = {"render_modes": ["human"]}
    PERF_LOW = -1.0
    PERF_HIGH = 1.0


    def __init__(self, env_config: Dict = None, **kwargs):
        super().__init__()
        if env_config is None:
            env_config = {}
        if kwargs:
            env_config = dict(env_config)
            env_config.update(kwargs)
        self.mode = env_config.get("mode", "train")
        self._env_config = dict(env_config)
        self.generalize = env_config.get("generalize", True)
        self.num_workers = int(env_config.get("num_workers", 0))
        default_engine = "matlab" if str(self.mode).lower() == "validate" else "mock"
        self.engine_name = str(env_config.get("engine", default_engine)).lower()
        self.csv_path = env_config.get(
            "csv_path",
            env_config.get("test_csv", "data/test_result.csv")
        )
        self.max_episode_steps = int(env_config.get("max_episode_steps", 5))
        self.spec = type(
            "spec",
            (),
            {"id": "ChaosMapEnv-v0", "max_episode_steps": self.max_episode_steps},
        )()
        config_path = env_config.get(
            "map_config_path",
            env_config.get(
                "config_path",
                os.path.join(os.path.dirname(__file__), "../configs/map_config_mscmi.yaml"),
            ),
        )
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.params = self._parse_params(self.cfg["params"])
        self.params_id = list(self.params.keys())
        self.fallback_target_metrics = dict(self.cfg["target_chaos_metrics"])
        self.target_metrics = dict(self.fallback_target_metrics)
        self.metric_ids = list(self.target_metrics.keys())
        self.norm_factors = np.array(
            self.cfg.get("normalize", [1.0] * len(self.metric_ids)),
            dtype=np.float64,
        )
        self.action_meaning = {0: -1, 1: 0, 2: +1}
        self.action_space = spaces.MultiDiscrete([3] * len(self.params_id))


        num_metrics = len(self.metric_ids)
        num_params = len(self.params_id)
        self._num_worst_corner_obs = 2
        obs_size = 2 * num_metrics + num_params + self._num_worst_corner_obs
        self.observation_space = spaces.Box(
            low=self.PERF_LOW,
            high=self.PERF_HIGH,
            shape=(obs_size,),
            dtype=np.float32,
        )
        engine_cfg = {
            "map_config_path": os.path.abspath(config_path),
            "config_path": config_path,
            "num_workers": self.num_workers,
            "mode": self.mode,
            "project_dir": env_config.get("project_dir", os.path.abspath(".")),
            "runs_base_dir": env_config.get("runs_base_dir", os.path.abspath("runs")),
            "templates_dir": env_config.get("templates_dir", os.path.abspath("templates")),
            "matlab_cmd": env_config.get("matlab_cmd", "matlab"),
            "chaotic_dir": env_config.get("chaotic_dir", "eval_engines/matlab"),
            "chaotic_func": env_config.get("chaotic_func", "chaotic"),
            "timeout_s": int(env_config.get("timeout_s", 600)),
            "default_csv_path": self.csv_path,
            "spectre_timeout": int(env_config.get("spectre_timeout", 900)),
            "ocean_timeout": int(env_config.get("ocean_timeout", 900)),
            "spectre_retries": int(env_config.get("spectre_retries", 1)),
            "lqtimeout": int(env_config.get("lqtimeout",
                             max(60, int(env_config.get("spectre_timeout", 900)) - 300))),
            "cache_wait_s": int(env_config.get("cache_wait_s", 500)),
        }
        self.engine = create_engine(self.engine_name, engine_cfg)
        self.cur_params_idx: Optional[np.ndarray] = None
        self.cur_chaos_metrics: Optional[Dict[str, float]] = None
        self.cur_corner_metrics = []
        self.pvt_reward_cfg = self.cfg.get('pvt_reward', {})
        self.prev_chaos_metrics: Optional[Dict[str, float]] = None
        self.episode_step = 0
        self.training_iter = 0
        self.current_target: Optional[Dict[str, float]] = None
        self.base_target_metrics: Optional[Dict[str, float]] = None
        self._nominal_metrics = {
            k: float(v) for k, v in self.cfg.get("nominal_metrics", {}).items()
            if k in ("ALE", "chaotic_ratio")
        }
        self._prev_step_reward = None
        print(f"[ChaosMapEnv] Engine: {self.engine_name}")
        print(f"[ChaosMapEnv] csv_path: {self.csv_path}")
        print(f"[ChaosMapEnv] max_episode_steps: {self.max_episode_steps}")


    def _parse_params(self, params_dict: Dict) -> Dict[str, np.ndarray]:
        parsed: Dict[str, np.ndarray] = {}
        for name, triple in params_dict.items():
            if len(triple) != 3:
                raise ValueError(f"Param '{name}' must be [min, max, step], got: {triple}")
            min_val, max_val, step = triple
            parsed[name] = np.arange(float(min_val), float(max_val) + float(step) / 2.0, float(step))
        return parsed


    def _cfg(self, key, default):
        if key in self._env_config:
            return self._env_config[key]
        if key in self.cfg:
            return self.cfg[key]
        pvt = self.cfg.get("pvt_reward", {}) or {}
        if key in pvt:
            return pvt[key]
        return default
    def _load_adaptive_base_targets(self) -> Dict[str, float]:
        fallback = dict(self.fallback_target_metrics)
        adapt_cfg = self.cfg.get("adaptive_target_config", {})
        if not adapt_cfg.get("enabled", False):
            return fallback
        cache_path = adapt_cfg.get("cache_path", "runs/metrics_cache.json")
        min_samples = int(adapt_cfg.get("min_samples", 40))
        mode = str(adapt_cfg.get("mode", "balanced_topk")).lower()
        top_k = int(adapt_cfg.get("top_k", 12))
        smoothing = float(adapt_cfg.get("smoothing", 0.70))
        relax_factor_cfg = adapt_cfg.get("relax_factor", {})
        floor_cfg = adapt_cfg.get("floor", {})
        ceiling_cfg = adapt_cfg.get("ceiling", {})
        score_weights = adapt_cfg.get("score_weights", {})
        if not os.path.isabs(cache_path):
            cache_path = os.path.abspath(cache_path)
        if not os.path.exists(cache_path):
            return fallback
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"[ChaosMapEnv] Adaptive targets: failed to read cache: {e}")
            return fallback
        if not isinstance(cache, dict) or len(cache) < min_samples:
            return fallback
        valid_rows = []
        for _, entry in cache.items():
            try:
                ale = float(entry.get("ALE", 0.0))
                cr = float(entry.get("chaotic_ratio", 0.0))
            except Exception:
                continue
            if np.isfinite(ale) and np.isfinite(cr):
                valid_rows.append({"ALE": ale, "chaotic_ratio": cr})
        if len(valid_rows) < min_samples:
            return fallback
        if mode != "balanced_topk":
            return fallback
        w_ale = float(score_weights.get("ALE", 3.5))
        w_cr = float(score_weights.get("chaotic_ratio", 5.0))
        ranked = sorted(valid_rows, key=lambda r: w_ale * r["ALE"] + w_cr * r["chaotic_ratio"], reverse=True)
        top_rows = ranked[:max(1, min(top_k, len(ranked)))]
        observed_targets = {
            "ALE_min": float(np.mean([r["ALE"] for r in top_rows])),
            "chaotic_ratio_min": float(np.mean([r["chaotic_ratio"] for r in top_rows])),
        }
        adapted = {}
        for metric_name, fallback_value in fallback.items():
            relax = float(relax_factor_cfg.get(metric_name, 1.0))
            observed_value = observed_targets.get(metric_name, fallback_value) * relax
            blended = (1.0 - smoothing) * float(fallback_value) + smoothing * float(observed_value)
            floor_val = float(floor_cfg.get(metric_name, -np.inf))
            ceil_val = float(ceiling_cfg.get(metric_name, np.inf))
            adapted[metric_name] = float(np.clip(blended, floor_val, ceil_val))
        print(
            f"[ChaosMapEnv] Adaptive base targets: "
            f"ALE_min={adapted.get('ALE_min', 0.0):.4f}, "
            f"CR_min={adapted.get('chaotic_ratio_min', 0.0):.4f}"
        )
        return adapted


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.base_target_metrics = self._load_adaptive_base_targets()
        if self.generalize and self.mode == "train":
            self.current_target = self._sample_random_target(self.base_target_metrics)
        else:
            self.current_target = dict(self.base_target_metrics)
        self.target_metrics = dict(self.base_target_metrics)
        lhs_pool_path = (
            self._env_config.get("lhs_pool_path")
            or self.cfg.get("lhs_pool_path", "runs/lhs_pool.json")
        )
        if os.path.exists(lhs_pool_path):
            with open(lhs_pool_path) as _f:
                _pool = json.load(_f)
                self._lhs_pool_cache = _pool
            if _pool:
                _entry = _pool[self.np_random.integers(0, len(_pool))]
                _p = _entry["params"]
                init_idx = []
                for name, vals in self.params.items():
                    if name in _p:
                        arr = np.array(vals)
                        idx = int(np.argmin(np.abs(arr - float(_p[name]))))
                    else:
                        idx = self.np_random.integers(0, len(vals))
                    init_idx.append(idx)
                self.cur_params_idx = np.array(init_idx, dtype=np.int32)
                print(f"[LHS] Initialized from pool (CR={_entry['CR_nominal']:.4f})")
            else:
                self.cur_params_idx = np.array(
                    [self.np_random.integers(0, len(v)) for v in self.params.values()],
                    dtype=np.int32,
                )
        else:
            self.cur_params_idx = np.array(
                [self.np_random.integers(0, len(v)) for v in self.params.values()],
                dtype=np.int32,
            )
        self._prev_step_reward = None
        self.cur_chaos_metrics = self._evaluate_design(self.cur_params_idx)
        self.prev_chaos_metrics = dict(self.cur_chaos_metrics)
        self.episode_step = 0
        return self._get_observation(), self._get_info()


    def step(self, action: np.ndarray):
        delta_max = int(self._cfg("adaptive_delta_max", 10))
        delta_min = int(self._cfg("adaptive_delta_min", 1))
        beta = float(self._cfg("adaptive_beta", 0.995))
        delta = max(delta_min, int(round(delta_max * (beta ** self.training_iter))))
        adj = np.array([self.action_meaning[int(a)] * delta for a in action], dtype=np.int32)
        self.cur_params_idx = np.clip(
            self.cur_params_idx + adj,
            [0] * len(self.params_id),
            [len(v) - 1 for v in self.params.values()],
        )
        self.prev_chaos_metrics = dict(self.cur_chaos_metrics)
        self.cur_chaos_metrics = self._evaluate_design(self.cur_params_idx)
        reward = self._compute_reward()
        terminated = False
        truncated = self._check_success(reward) or (self.episode_step >= (self.max_episode_steps - 1))
        self.episode_step += 1
        return self._get_observation(), reward, terminated, truncated, self._get_info()


    def _is_param_combo_safe(self, param_values):
        try:
            for name, vals in self.params.items():
                w = float(param_values.get(name, vals[0]))
                if w < vals[0]: return False, f'{name} below min {vals[0]*1e9:.0f}nm'
                if w > vals[-1]: return False, f'{name} above max {vals[-1]*1e6:.1f}um'
            return True, 'ok'
        except Exception as e:
            return False, f'safety-check exception: {e}'


    def _evaluate_corner(self, param_values: dict, vdd: float, temp: float, process: str = 'tt') -> dict:
        import traceback
        corner_params = dict(param_values)
        corner_params['VDD'] = vdd
        corner_params['TEMP'] = temp
        corner_params['PROCESS'] = process
        try:
            return self.engine.evaluate(corner_params)
        except SpectreSimulationError:
            return {'MLE': 0.0, 'ALE': 0.0, 'chaotic_ratio': 0.0,
                    'bifurcation_density': 0.0, 'power_mw': 0.0, 'area_um2': 0.0,
                    'sim_failed': True}
        except Exception as e:
            traceback.print_exc()
            print(f'[ChaosMapEnv] Corner ({vdd}V,{temp}C) error: {e}')
            return {'MLE': 0.0, 'ALE': 0.0, 'chaotic_ratio': 0.0,
                    'bifurcation_density': 0.0, 'power_mw': 0.0, 'area_um2': 0.0,
                    'sim_failed': True}


    def _evaluate_design(self, param_idx: np.ndarray) -> Dict[str, float]:
        param_values = OrderedDict()
        for i, name in enumerate(self.params_id):
            param_values[name] = self.params[name][param_idx[i]]
        safe_ok, safe_reason = self._is_param_combo_safe(param_values)
        if not safe_ok:
            print(f"[ChaosMapEnv] Pre-Spectre reject: {safe_reason}")
            zero = {'MLE': 0.0, 'ALE': 0.0, 'chaotic_ratio': 0.0,
                    'bifurcation_density': 0.0, 'power_mw': 999.0, 'area_um2': 999.0}
            self.cur_corner_metrics = [zero]
            self._progressive_fast_penalty = True
            self._progressive_cr_nom = 0.0
            self._progressive_C_pen = float(self._cfg('progressive_penalty', 0.1))
            return zero
        try:
            if self.engine_name in ("matlab", "chaotic"):
                raw = self.engine.evaluate(param_values, csv_path=getattr(self, "csv_path", None))
                return {
                    "MLE": raw.get("MLE", 0.0),
                    "ALE": raw.get("ALE", 0.0),
                    "chaotic_ratio": raw.get("chaotic_ratio", 0.0),
                    "bifurcation_density": 0.0,
                    "power_mw": 0.0,
                    "area_um2": 0.0,
                }
            else:
                # pvt_corners from map config; DEFAULT_CORNERS as fallback
                pvt_corners_cfg = self.cfg.get('pvt_corners', [])
                if pvt_corners_cfg:
                    corners = [
                        (float(c['VDD']), float(c['temp']), c['process'], c['process'])
                        for c in pvt_corners_cfg
                    ]
                else:
                    from autochaos.envs.reward_pvt2 import DEFAULT_CORNERS
                    print('[ChaosMapEnv] WARNING: map config has no pvt_corners section - '
                          'using built-in default corners. Define pvt_corners for your circuit.')
                    corners = [(v, t, name, proc) for (v, t, name, proc) in DEFAULT_CORNERS]
                C_pen = float(self._cfg('progressive_penalty', 0.1))
                progressive_tau = float(
                    self.cfg.get('pvt_reward', {}).get('progressive_tau',
                    self.cfg.get('progressive_tau', 0.25))
                )
                nom_vdd, nom_temp, nom_cname, nom_process = corners[0]
                cm_nom = self._evaluate_corner(param_values, nom_vdd, nom_temp, nom_process)
                print('[ChaosMapEnv] Corner ' + nom_cname + ' (' + str(nom_vdd) + 'V,' + str(nom_temp) + 'C): '
                      + 'ALE=' + str(round(cm_nom.get('ALE', 0.0), 4))
                      + ' CR=' + str(round(cm_nom.get('chaotic_ratio', 0.0), 4)))
                cr_nom = cm_nom.get('chaotic_ratio', 0.0)
                if cr_nom < progressive_tau:
                    print('[ChaosMapEnv] Progressive gate: CR_nom=' + str(round(cr_nom,4))
                          + ' < ' + str(progressive_tau) + ' - skipping SS/FF')
                    self.cur_corner_metrics = [cm_nom]
                    self._progressive_fast_penalty = True
                    self._progressive_cr_nom = cr_nom
                    self._progressive_C_pen = C_pen
                    return cm_nom
                else:
                    self._progressive_fast_penalty = False
                    corner_results = [cm_nom]
                    for vdd, temp, cname, process in corners[1:]:
                        cm = self._evaluate_corner(param_values, vdd, temp, process)
                        corner_results.append(cm)
                        print('[ChaosMapEnv] Corner ' + cname + ' (' + str(vdd) + 'V,' + str(temp) + 'C): '
                              + 'ALE=' + str(round(cm.get('ALE', 0.0), 4))
                              + ' CR=' + str(round(cm.get('chaotic_ratio', 0.0), 4)))
                    self.cur_corner_metrics = corner_results
                    return corner_results[0]
        except SpectreSimulationError:
            print("[ChaosMapEnv] Spectre failed (bad params) - returning zero metrics")
            return self._design_level_failure()
        except Exception as e:
            print(f"[ChaosMapEnv] Simulation failed: {e}")
            return self._design_level_failure()
    def _design_level_failure(self):
        zero = {'MLE': 0.0, 'ALE': 0.0, 'chaotic_ratio': 0.0,
                'bifurcation_density': 0.0, 'power_mw': 999.0, 'area_um2': 999.0,
                'sim_failed': True}
        self.cur_corner_metrics = [zero]
        self._progressive_fast_penalty = True
        self._progressive_cr_nom = 0.0
        self._progressive_C_pen = float(self._cfg('progressive_penalty', 0.1))
        return zero


    def _compute_reward(self) -> float:
        from autochaos.envs.reward_pvt2 import compute_pvt_reward, format_pvt_log, DEFAULT_TAU_ALE, DEFAULT_TAU_CR
        if getattr(self, "_progressive_fast_penalty", False):
            cr_nom = getattr(self, "_progressive_cr_nom", 0.0)
            C_pen = getattr(self, "_progressive_C_pen", 0.1)
            cfg_pvt = getattr(self, "pvt_reward_cfg", {})
            w_cr = float(cfg_pvt.get("w_CR", 0.5))
            fast_reward = w_cr * (cr_nom ** 1.5) - C_pen
            print(f"[RewardPVT] Progressive fast reject: CR_nom={cr_nom:.4f} => REWARD={fast_reward:.4f}")
            return fast_reward
        corner_metrics = getattr(self, 'cur_corner_metrics', None) or [self.cur_chaos_metrics]
        cfg_pvt = getattr(self, 'pvt_reward_cfg', {})
        _corner_names = [
            (c.get('name') or c.get('process') or f'corner{i}') if isinstance(c, dict) else f'corner{i}'
            for i, c in enumerate(self.cfg.get('pvt_corners', []) or [])
        ]
        if _corner_names and 'name' not in (self.cfg.get('pvt_corners') or [{}])[0]:
            _corner_names[0] = 'nominal'
        reward, dbg = compute_pvt_reward(
            corner_metrics,
            tau_ale=float(cfg_pvt.get('tau_ALE', DEFAULT_TAU_ALE)),
            tau_cr=float(cfg_pvt.get('tau_CR', DEFAULT_TAU_CR)),
            w_ale=float(cfg_pvt.get('w_ALE', 0.5)),
            w_cr=float(cfg_pvt.get('w_CR', 0.5)),
            alpha=float(cfg_pvt.get('alpha', 0.6)),
            bonus=float(cfg_pvt.get('bonus', 2.0)),
            penalty=float(cfg_pvt.get('penalty', 1.0)),
            robust_credit=float(cfg_pvt.get('robust_credit', 1.0)),
            corner_names=_corner_names or None,
        )
        print(f'[RewardPVT] {format_pvt_log(dbg)}')
        param_values = {name: self.params[name][self.cur_params_idx[i]]
                        for i, name in enumerate(self.params_id)}
        A_max = float(self._cfg("A_max", 1e6))
        lambda_a = float(self._cfg("lambda_a", 0.0))
        area_um2 = self._compute_area_um2(param_values)
        P_area = lambda_a * max(0.0, area_um2 - A_max)
        if P_area > 0:
            print(f"[ConstraintReward] Area={area_um2:.4f}um2 > A_max={A_max} => P_area={P_area:.4f}")
        return reward - P_area


    def _check_success(self, reward: float) -> bool:
        corner_metrics = getattr(self, 'cur_corner_metrics', None)
        if not corner_metrics or getattr(self, '_progressive_fast_penalty', False):
            return False
        cfg_pvt = getattr(self, 'pvt_reward_cfg', {})
        from autochaos.envs.reward_pvt2 import DEFAULT_TAU_ALE, DEFAULT_TAU_CR
        tau_ale = float(cfg_pvt.get('tau_ALE', DEFAULT_TAU_ALE))
        tau_cr = float(cfg_pvt.get('tau_CR', DEFAULT_TAU_CR))
        for cm in corner_metrics:
            if cm.get('ALE', 0.0) < tau_ale or cm.get('chaotic_ratio', 0.0) < tau_cr:
                return False
        print(f"[ChaosMapEnv] SUCCESS — all_pass=True at step {self.episode_step}, terminal bonus applied")
        return True


    def _compute_area_um2(self, param_values: dict) -> float:
        fixed_lengths = {k: float(v) for k, v in self.cfg.get("fixed_lengths", {}).items()}
        if not fixed_lengths:
            return 0.0
        area = 0.0
        for name, L in fixed_lengths.items():
            W = float(param_values.get(name, list(self.params.get(name, [120e-9]))[0]))
            area += W * L
        return area * 1e12


    def _get_observation(self) -> np.ndarray:
        current_values = np.array(
            [self.cur_chaos_metrics.get(metric.replace("_min", "").replace("_max", ""), 0.0)
             for metric in self.metric_ids],
            dtype=np.float64,
        )
        target_values = np.array(list(self.current_target.values()), dtype=np.float64)
        current_norm = normalize_metrics(current_values, self.norm_factors)
        deficits = np.array(
            [(target - current) / (target + 1e-12)
             for current, target in zip(current_values, target_values)],
            dtype=np.float64,
        )
        deficits = np.clip(deficits, -1.0, 1.0)
        params_norm = np.array(
            [idx / (len(self.params[name]) - 1)
             for idx, name in zip(self.cur_params_idx, self.params_id)],
            dtype=np.float64,
        )
        corner_metrics = getattr(self, 'cur_corner_metrics', None) or [self.cur_chaos_metrics]
        min_cr  = min(cm.get('chaotic_ratio', 0.0) for cm in corner_metrics)
        min_ale = min(cm.get('ALE', 0.0) for cm in corner_metrics)
        cfg_pvt  = getattr(self, 'pvt_reward_cfg', {})
        tau_cr   = float(cfg_pvt.get('tau_CR',  0.95))
        tau_ale  = float(cfg_pvt.get('tau_ALE', 0.35))
        worst_corner_norm = np.clip(
            np.array([min_cr / max(tau_cr, 1e-9), min_ale / max(tau_ale, 1e-9)],
                     dtype=np.float64),
            self.PERF_LOW, self.PERF_HIGH
        )
        obs = np.concatenate([current_norm, deficits, params_norm, worst_corner_norm])
        obs = np.clip(obs, self.PERF_LOW, self.PERF_HIGH)
        return obs.astype(np.float32)


    def _get_info(self) -> Dict:
        corner_metrics = getattr(self, 'cur_corner_metrics', None) or [self.cur_chaos_metrics]
        cfg_pvt  = getattr(self, 'pvt_reward_cfg', {})
        tau_cr   = float(cfg_pvt.get('tau_CR',  0.95))
        tau_ale  = float(cfg_pvt.get('tau_ALE', 0.35))
        all_pass = all(
            cm.get('chaotic_ratio', 0.0) >= tau_cr and cm.get('ALE', 0.0) >= tau_ale
            for cm in corner_metrics
        ) and not getattr(self, '_progressive_fast_penalty', False)
        param_values = {
            name: self.params[name][idx]
            for name, idx in zip(self.params_id, self.cur_params_idx)
        }
        area_um2 = self._compute_area_um2(param_values)
        return {
            "episode_step": self.episode_step,
            "current_metrics": dict(self.cur_chaos_metrics),
            "corner_metrics": [dict(cm) for cm in corner_metrics],
            "min_cr": min(cm.get('chaotic_ratio', 0.0) for cm in corner_metrics),
            "min_ale": min(cm.get('ALE', 0.0) for cm in corner_metrics),
            "all_pass": all_pass,
            "area_um2": area_um2,
            "target_metrics": dict(self.current_target),
            "base_target_metrics": dict(self.base_target_metrics) if self.base_target_metrics is not None else {},
            "param_values": param_values,
        }


    def _sample_random_target(self, base_target: Dict[str, float]) -> Dict[str, float]:
        gen_cfg = self.cfg.get("target_generalization", {})
        enabled = bool(gen_cfg.get("enabled", True))
        low = float(gen_cfg.get("low", 0.95))
        high = float(gen_cfg.get("high", 1.05))
        if not enabled:
            return dict(base_target)
        return {
            metric_name: float(base_value) * np.random.uniform(low, high)
            for metric_name, base_value in base_target.items()
        }


    def close(self):
        if hasattr(self, "engine"):
            try:
                self.engine.close()
            except Exception:
                pass


gym.register(

    id="chaos-map-v0",
    entry_point="autochaos.envs.chaos_map_env:ChaosMapEnv",

)

