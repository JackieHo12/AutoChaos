# How AutoChaos Works

This is the tour I wish I had when I started. It follows one training step end
to end and then explains where everything lives. The thesis covers the theory.
This document covers the code.

## One step, end to end

Training is a loop between RLlib and the circuit simulator. RLlib owns the
outer loop. It holds the policy network (a small MLP), asks it for actions,
and updates it from collected batches. Everything circuit-related happens
inside `ChaosMapEnv` (`autochaos/envs/chaos_map_env.py`), which is a standard
Gymnasium environment, so RLlib does not know or care that stepping the
environment takes seconds of SPICE time.

A step goes like this. The env holds the current design, a vector of widths
and lengths on the grid defined in the map config. The action from the policy
is one move per parameter: down a step, hold, or up a step. The env applies
the moves, clamps to the parameter bounds, and hands the new design to the
engine.

The engine (`eval_engines/ngspice/ngspice_engine.py` or
`eval_engines/cadence/cadence_engine.py`) turns the design into a netlist by
filling the template's placeholders, runs the simulator, and parses the
transfer curves back out. NGSpice does this as 551 small independent sweeps
run by a process pool. Spectre does it as one nested parametric sweep followed
by an OCEAN export. Either way the result is the same thing: output voltage
as a function of input voltage, for each slice of the control-voltage sweep.

The chaos analysis then iterates each transfer curve as a map, accumulates
log-slopes along the trajectory, and produces a Lyapunov exponent per sweep
point. CR is the fraction of sweep points with a positive exponent and ALE is
the average exponent over just those points. This happens per corner.

The reward (`autochaos/envs/reward_pvt2.py`) is built from the corner results.
Each metric is scored against its threshold, the two scores are mixed with the
configured weights, and the corner scores are collapsed with the worst-corner
blend (alpha times the minimum plus one minus alpha times the mean). Designs
that pass both thresholds at every corner get the fixed all-pass bonus on top,
which is why good designs in the results sit near 2.5 instead of 0.5. Designs
that are not chaotic enough at nominal never get corner simulations at all.
The gate below explains why.

The reward goes back to RLlib, the episode continues for up to five steps,
and after enough steps have been collected across all workers RLlib performs
one PPO update. The policy is fixed while a batch is being collected.
Improvement happens between iterations, not inside them.

## The progressive gate

Corner simulations are the expensive part, three full sweeps instead of one.
The env therefore evaluates the nominal corner first and only spends the SS
and FF simulations on designs whose nominal CR clears the progressive gate
(0.25 for the 3T configs, 0.10 for MSCMI). Early in a run almost everything
is rejected at the gate and that is the intended behavior. The reward still
slopes upward with nominal CR in that regime, so the policy learns which way
is up without paying for corners on hopeless designs.

## The cache

Every simulated design lands in `metrics_cache.db` (SQLite), keyed by the
exact parameter values plus corner plus model. Because actions move on a
grid, revisits are exact hits and cost nothing. Two consequences worth
knowing. First, the cache is also the run's archive. Every analysis script
in `scripts/` works from it, and the tables in the thesis were built from it.
Second, the key does not include which run produced it, so if you switch
device model files you must start with a fresh cache. Otherwise stale
metrics from the old model will be served for matching geometries.

## The warm start

Before training, `lhs_sampler.py` builds a pool of starting designs. It draws
stratified samples across the parameter box (Latin hypercube, 500 samples for
the 3T configs), injects the hand-sized nominal design so at least one known
chaotic ancestor is present, evaluates the pool, and keeps the best 50 as
`runs/lhs_pool.json`. Every episode reset draws from this pool. The pool
never changes during training. What changes is what the policy does from
those starting points.

## Where the knobs live

The map config (`autochaos/configs/map_config_*.yaml`) is the circuit: the
template path, the tunable parameters with their bounds and grid steps, the
nominal design, the sweep definition, the corner list, and the reward
constants (thresholds, weights, alpha, gate). The training config
(`training_config_*.yaml`) is the run: which map config, which engine
binary, worker counts, timeouts, cache behavior. If you are changing what is
being optimized you are in the map config. If you are changing how the run
executes, you are in the training config. The iteration count is neither. It
comes from `--episodes` on the command line.

## What lands on disk

`runs/` holds per-evaluation working directories (deleted after parsing, with
only the CSV surviving on NGSpice), the LHS pool, and the Ray results
directory with `progress.csv` and `result.json` per training iteration.
`metrics_cache.db` accumulates at the package root. The training log you
tee'd is the best record of what actually happened. The analysis scripts
cross-reference it with the cache.

## Reading a run

The mean episode return in `progress.csv` starts near the reject floor and
climbs as the policy learns to clear the gate. Do not expect it to reach the
all-pass region. It is an average over mostly-hard episodes, and a run can be
finding all-pass designs steadily while the mean sits below zero. Count
all-pass designs in the cache (scripts do this) rather than reading them off
the return curve.
