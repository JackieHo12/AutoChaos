import argparse
import os
import sys
import time
import yaml


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autochaos.envs.chaos_map_env import ChaosMapEnv


CHECKPOINT_DIR = os.path.abspath("runs/checkpoints")

CHECKPOINT_FREQ = 10


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="autochaos/configs/training_config.yaml")
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "validate", "baseline"])
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def run_validate(env_config, episodes):

    print("\n" + "=" * 60)
    print("VALIDATE MODE (no Ray - direct env loop)")
    print("=" * 60)
    total_start = time.perf_counter()
    env = ChaosMapEnv(env_config)
    for ep in range(episodes):
        ep_start = time.perf_counter()
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated or (steps >= env.max_episode_steps)
        ep_elapsed = time.perf_counter() - ep_start
        print(f"Episode {ep+1}/{episodes} reward={total_reward:.3f} steps={steps} elapsed={ep_elapsed:.1f}s")
    env.close()
    total_elapsed = time.perf_counter() - total_start
    print(f"[train.py] TOTAL VALIDATE ELAPSED = {total_elapsed:.1f}s")


def _sweep_orphan_pending(runs_dir="runs"):
    import sqlite3
    db_path = os.path.join(runs_dir, "metrics_cache.db")
    if not os.path.exists(db_path):
        return
    try:
        con = sqlite3.connect(db_path, timeout=30.0)
        cur = con.execute("DELETE FROM cache WHERE pending=1")
        con.commit()
        n = cur.rowcount
        con.close()
        print(f"[train.py] Orphan sweep: removed {n} stale pending cache row(s)")
    except Exception as e:
        print(f"[train.py] Orphan sweep warning: {e}")


def run_training(env_config, train_cfg, args):

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from autochaos.callbacks import AutoChaosCallbacks
    from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
    from ray.rllib.connectors.env_to_module import MeanStdFilter
    from ray.tune.registry import register_env
    register_env("chaos-map-v0", lambda cfg: ChaosMapEnv(cfg))
    num_workers = int(train_cfg.get("num_workers", 0))
    num_envs_per_worker = int(train_cfg.get("num_envs_per_worker", 1))
    rfl_raw = train_cfg.get("rollout_fragment_length", 5)
    rollout_fragment_length = rfl_raw if rfl_raw == "auto" else int(rfl_raw)
    sample_timeout_s = float(train_cfg.get("sample_timeout_s", 14400.0))
    ray_cpus = max(2, num_workers * num_envs_per_worker + 2)
    print(f"[train.py] FINAL train_cfg.num_workers = {num_workers}")
    print(f"[train.py] FINAL env_config.engine = {env_config.get('engine')}")
    print(f"[train.py] FINAL env_config.num_workers = {env_config.get('num_workers')}")
    print(f"[train.py] Parallel rollout config: workers={num_workers}, envs_per_worker={num_envs_per_worker}, rollout_fragment_length={rollout_fragment_length}, sample_timeout_s={sample_timeout_s}")
    print("[train.py] Registering custom env: chaos-map-v0")
    _sweep_orphan_pending(env_config.get("runs_base_dir", "runs"))
    total_start = time.perf_counter()
    _seed = train_cfg.get("seed", None)
    ray.init(ignore_reinit_error=True, num_cpus=ray_cpus, num_gpus=0)
    entropy_coeff_schedule = train_cfg.get("entropy_coeff_schedule", None)
    grad_clip_val = train_cfg.get("grad_clip", None)
    grad_clip = float(grad_clip_val) if grad_clip_val is not None else None
    _obs_filter = train_cfg.get("observation_filter", "MeanStdFilter")
    if _obs_filter == "MeanStdFilter":
        env_to_module_connector = lambda env, spaces=None, device=None: MeanStdFilter()
    else:
        env_to_module_connector = None
    _model_cfg = train_cfg.get("model", {})
    _fcnet_hiddens   = _model_cfg.get("fcnet_hiddens", [128, 128])
    _fcnet_activation = _model_cfg.get("fcnet_activation", "relu")
    _vf_share_layers  = bool(_model_cfg.get("vf_share_layers", False))
    model_config = DefaultModelConfig(
        fcnet_hiddens=_fcnet_hiddens,
        fcnet_activation=_fcnet_activation,
        vf_share_layers=_vf_share_layers,
    )
    cfg = (
        PPOConfig()
        .environment(env="chaos-map-v0", env_config=env_config, disable_env_checking=True)
        .training(
            train_batch_size_per_learner=int(train_cfg.get("train_batch_size_per_learner", train_cfg.get("train_batch_size", 100))),
            minibatch_size=int(train_cfg.get("sgd_minibatch_size", 40)),
            num_epochs=int(train_cfg.get("num_sgd_iter", 10)),
            lr=float(train_cfg.get("lr", 3e-4)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            lambda_=float(train_cfg.get("lambda", 0.95)),
            clip_param=float(train_cfg.get("clip_param", 0.2)),
            kl_coeff=float(train_cfg.get("kl_coeff", 0.3)),
            kl_target=float(train_cfg.get("kl_target", 0.01)),
            vf_clip_param=float(train_cfg.get("vf_clip_param", 3.0)),
            vf_loss_coeff=float(train_cfg.get("vf_loss_coeff", 0.5)),
            entropy_coeff=entropy_coeff_schedule if entropy_coeff_schedule is not None else float(train_cfg.get("entropy_coeff", 0.01)),
            grad_clip=grad_clip,
        )
        .env_runners(
            num_env_runners=num_workers,
            num_envs_per_env_runner=num_envs_per_worker,
            rollout_fragment_length=rollout_fragment_length,
            batch_mode=train_cfg.get("batch_mode", "complete_episodes"),
            sample_timeout_s=sample_timeout_s,
            env_to_module_connector=env_to_module_connector,
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .rl_module(model_config=model_config)
        .fault_tolerance(
            restart_failed_env_runners=True,
            max_num_env_runner_restarts=100,
            delay_between_env_runner_restarts_s=30.0,  # 100 x 30s = ~50min self-heal budget
        )
        .env_runners(
            validate_env_runners_after_construction=(
                train_cfg.get("validate_env_runners_after_construction", False)
            ),
        )
        .callbacks(AutoChaosCallbacks)
        .resources(num_gpus=0)
        .framework("torch")
        .debugging(seed=_seed)
    )
    algo = cfg.build_algo()
    print("[train.py] PPO built OK")
    if args.restore:
        algo.restore_from_path(args.restore)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_reward = float("-inf")
    STALL_EXIT_S = 3 * 3600.0
    last_timesteps = -1
    last_progress_time = time.time()
    effective_iter = 0
    for i in range(args.episodes):
        iter_start = time.perf_counter()
        _clock = effective_iter
        algo.env_runner_group.foreach_env_runner(lambda w: setattr(w.env.envs[0] if hasattr(w.env, "envs") else w.env, "training_iter", _clock) if hasattr(w, "env") and w.env is not None else None)
        result = algo.train()
        iter_elapsed = time.perf_counter() - iter_start
        env_results = result.get("env_runners", {})
        reward = env_results.get("episode_return_mean", float("nan"))
        ep_len = env_results.get("episode_len_mean", float("nan"))
        ep_max = env_results.get("episode_return_max", float("nan"))
        timesteps = result.get("num_env_steps_sampled_lifetime", 0)
        print(f"Iter {i+1}/{args.episodes}: reward={reward:.3f}, reward_max={ep_max:.3f}, ep_len={ep_len:.1f}, timesteps={timesteps}, iter_elapsed={iter_elapsed:.1f}s")
        if timesteps and timesteps != last_timesteps:
            last_timesteps = timesteps
            last_progress_time = time.time()
            effective_iter += 1
        else:
            stalled_s = time.time() - last_progress_time
            print(f"[train.py] WATCHDOG: no new env samples this iteration (stalled {stalled_s/60.0:.1f} min, effective_iter={effective_iter})")
            if stalled_s > STALL_EXIT_S:
                stall_path = os.path.join(CHECKPOINT_DIR, "stall_exit")
                algo.save_to_path(stall_path)
                print(f"[train.py] WATCHDOG: no samples for {STALL_EXIT_S/3600.0:.1f}h - env runners appear permanently dead (server resource storm?). Checkpoint: {stall_path}. Relaunch with --restore {stall_path} when the server recovers.")
                break
            time.sleep(60)
            continue
        if not (reward != reward):
            if reward > best_reward:
                best_reward = reward
                best_path = os.path.join(CHECKPOINT_DIR, "best")
                algo.save_to_path(best_path)
                print(f"[train.py] New best reward={reward:.3f} — checkpoint saved to {best_path}")
        if (i + 1) % CHECKPOINT_FREQ == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"iter_{i+1:04d}")
            algo.save_to_path(ckpt_path)
            print(f"[train.py] Checkpoint saved to {ckpt_path}")
    final_path = os.path.join(CHECKPOINT_DIR, "final")
    algo.save_to_path(final_path)
    total_elapsed = time.perf_counter() - total_start
    print(f"[train.py] TOTAL TRAIN ELAPSED = {total_elapsed:.1f}s")
    print(f"Done. Final checkpoint: {final_path}")
    algo.stop()
    ray.shutdown()


def main():

    args = parse_args()
    with open(args.config, "r") as f:
        train_cfg = yaml.safe_load(f)
    print(f"[train.py] Loaded config file: {args.config}")
    print(f"[train.py] Raw top-level num_workers from file: {train_cfg.get('num_workers')}")
    print(f"[train.py] Raw env_config from file: {train_cfg.get('env_config', {})}")
    env_config = train_cfg.get("env_config", {})
    env_config["mode"] = args.mode
    env_config["project_dir"] = os.path.abspath(os.getcwd())
    env_config["runs_base_dir"] = os.path.join(env_config["project_dir"], "runs")
    env_config["templates_dir"] = os.path.join(env_config["project_dir"], "templates")
    if args.engine is not None:
        env_config["engine"] = args.engine
    if args.workers is not None:
        train_cfg["num_workers"] = args.workers
    env_config["num_workers"] = int(train_cfg.get("num_workers", env_config.get("num_workers", 0)))
    print(f"[train.py] mode={args.mode}, engine={env_config.get('engine', 'mock')}")
    print(f"[train.py] project_dir={env_config['project_dir']}")
    if args.mode in ("validate", "baseline"):
        run_validate(env_config, episodes=max(args.episodes, 2))
    else:
        lhs_pool_path = env_config.get("lhs_pool_path", "runs/lhs_pool.json")
        if not os.path.exists(lhs_pool_path):
            print(f"[train.py] LHS pool not found at {lhs_pool_path} - running LHS sampler first...")
            from autochaos.lhs_sampler import run_lhs_sampler
            map_config_path = env_config.get("map_config_path")
            if not map_config_path:
                raise ValueError("lhs_pool_path set but map_config_path not found in env_config")
            M = int(env_config.get("lhs_M", 250))
            N = int(env_config.get("lhs_N", 20))
            n_workers = int(env_config.get("lhs_workers", 5))
            run_lhs_sampler(args.config, map_config_path, M=M, N=N, pool_path=lhs_pool_path, n_workers=n_workers)
        else:
            print(f"[train.py] LHS pool found at {lhs_pool_path} - skipping LHS sampling")
        run_training(env_config, train_cfg, args)


if __name__ == "__main__":

    main()

