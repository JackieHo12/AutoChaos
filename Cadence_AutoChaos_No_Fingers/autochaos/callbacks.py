import json
import os
import time
from ray.rllib.callbacks.callbacks import RLlibCallback


class AutoChaosCallbacks(RLlibCallback):
    def on_algorithm_init(self, *, algorithm, metrics_logger=None, **kwargs):
        try:
            runs_dir = algorithm.config.env_config.get('runs_base_dir', 'runs')
            os.makedirs(runs_dir, exist_ok=True)
            config_path = os.path.join(runs_dir, 'run_config.json')
            config_data = {
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'env_config': dict(algorithm.config.env_config),
                'num_env_runners': algorithm.config.num_env_runners,
                'train_batch_size_per_learner': algorithm.config.train_batch_size_per_learner,
                'gamma': algorithm.config.gamma,
                'lr': algorithm.config.lr,
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            print('[AutoChaosCallbacks] Run config saved to ' + config_path)
        except Exception as e:
            print('[AutoChaosCallbacks] on_algorithm_init warning: ' + str(e))

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        if metrics_logger is None:
            return
        try:
            info = episode.get_infos(-1) if hasattr(episode, 'get_infos') else {}
            if not isinstance(info, dict):
                info = {}
            current  = info.get('current_metrics', {})
            cr_nom   = float(current.get('chaotic_ratio', 0.0))
            ale_nom  = float(current.get('ALE', 0.0))
            min_cr   = float(info.get('min_cr',  cr_nom))
            min_ale  = float(info.get('min_ale', ale_nom))
            all_pass = bool(info.get('all_pass', False))
            area_um2 = float(info.get('area_um2', 0.0))
            metrics_logger.log_value(
                ('chaos', 'cr_nominal'), cr_nom, reduce='max', window=100,
            )
            metrics_logger.log_value(
                ('chaos', 'cr_nominal_iter'), cr_nom, reduce='max', window=20,
            )
            metrics_logger.log_value(
                ('chaos', 'ale_nominal'), ale_nom, reduce='max', window=100,
            )
            metrics_logger.log_value(
                ('chaos', 'cr_worst'), min_cr, reduce='max', window=100,
            )
            metrics_logger.log_value(
                ('chaos', 'cr_worst_iter'), min_cr, reduce='max', window=20,
            )
            metrics_logger.log_value(
                ('chaos', 'ale_worst'), min_ale, reduce='max', window=100,
            )
            metrics_logger.log_value(
                ('chaos', 'all_pass_rate'), 1.0 if all_pass else 0.0,
                reduce='mean', window=100,
            )
            if area_um2 > 0.0:
                metrics_logger.log_value(
                    ('chaos', 'area_um2'), area_um2, reduce='mean', window=100,
                )
        except Exception as e:
            print('[AutoChaosCallbacks] on_episode_end warning: ' + str(e))

    def on_env_runners_recreated(
        self, *, algorithm, env_runner_group,
        env_runner_indices, is_evaluation, **kwargs
    ):
        n = len(env_runner_indices)
        label = 'eval' if is_evaluation else 'train'
        print(
            '[AutoChaosCallbacks] ' + str(n) + ' ' + label +
            ' EnvRunner(s) recreated: indices=' + str(env_runner_indices)
        )
        try:
            if hasattr(algorithm, 'metrics') and algorithm.metrics is not None:
                algorithm.metrics.log_value(
                    ('fault', 'workers_recreated'),
                    float(n),
                    reduce='sum',
                )
        except Exception as e:
            print('[AutoChaosCallbacks] on_env_runners_recreated warning: ' + str(e))

    def _apply_kl_floor(self, algorithm, result):
        try:
            floor = float(algorithm.config.env_config.get('kl_coeff_min', 0.0) or 0.0)
        except Exception:
            floor = 0.0
        try:
            floor = float(getattr(algorithm, '_kl_coeff_min', floor) or floor)
        except Exception:
            pass
        if floor <= 0.0:
            return
        clamped = None
        try:
            lg = algorithm.learner_group
            learner = lg._learner if hasattr(lg, '_learner') else None
            if learner is not None and hasattr(learner, 'curr_kl_coeffs_per_module'):
                for mid, var in learner.curr_kl_coeffs_per_module.items():
                    cur = float(var.numpy()) if hasattr(var, 'numpy') else float(var)
                    if cur < floor:
                        try:
                            var.assign(floor)  # tf-style variable
                        except Exception:
                            import torch
                            with torch.no_grad():
                                var.fill_(floor)  # torch tensor
                        clamped = (mid, cur, floor)
        except Exception as e:
            if result is not None:
                result['kl_floor_note'] = 'clamp skipped: ' + str(e)
            return
        if clamped is not None:
            print('[KLFloor] kl_coeff raised ' + str(round(clamped[1], 5))
                  + ' -> ' + str(clamped[2]) + ' (module ' + str(clamped[0]) + ')')
            if result is not None:
                result['kl_coeff_floored_to'] = clamped[2]

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        self._apply_kl_floor(algorithm, result)
        try:
            env_cfg = algorithm.config.env_config
            map_cfg_path = env_cfg.get('map_config_path', '')
            if not map_cfg_path or not os.path.exists(map_cfg_path):
                return
            import yaml
            with open(map_cfg_path) as f:
                map_cfg = yaml.safe_load(f)
            curriculum = map_cfg.get('curriculum', {})
            if not curriculum.get('enabled', False):
                return
            stages = curriculum.get('stages', [])
            if not stages:
                return
            mean_reward = result.get('env_runners', {}).get('episode_return_mean', float('nan'))
            if mean_reward != mean_reward:
                return
            target_tau = None
            for stage in stages:
                if mean_reward >= float(stage.get('reward_threshold', float('inf'))):
                    target_tau = float(stage.get('progressive_tau'))
            if target_tau is None:
                return
            def _set_tau(worker):
                try:
                    env = (worker.env.envs[0]
                           if hasattr(worker.env, 'envs')
                           else worker.env)
                    if hasattr(env, 'cfg'):
                        old = env.cfg.get('progressive_tau', 0.0)
                        if abs(old - target_tau) > 1e-6:
                            env.cfg['progressive_tau'] = target_tau
                            print('[Curriculum] progressive_tau: '
                                  + str(round(old, 3)) + ' -> ' + str(target_tau))
                except Exception:
                    pass
            algorithm.env_runner_group.foreach_env_runner(_set_tau)
        except Exception as e:
            print('[AutoChaosCallbacks] on_train_result warning: ' + str(e))
