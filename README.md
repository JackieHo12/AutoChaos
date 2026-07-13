# AutoChaos

AutoChaos sizes the transistors of analog chaotic circuits with reinforcement
learning. A PPO agent (Ray RLlib) proposes widths and lengths, every candidate
is simulated across process, voltage and temperature corners, and the reward is
built from the worst corner rather than the nominal one, so the search is
pushed toward PVT-robust designs. Chaos is measured from the simulated
transfer characteristic: CR is the fraction of the control-voltage sweep with a
positive Lyapunov exponent, and ALE is the average Lyapunov exponent over that
sweep.

There are four self-contained packages, one per backend and device treatment:

* `Cadence_AutoChaos_No_Fingers` runs Cadence Spectre with plain W/L devices.
  This is the package the 3T Cadence run and the MSCMI runs used.
* `Cadence_AutoChaos_Fingers` is the Spectre backend with finger-aware sizing:
  the finger count of each device is derived from its width at a fixed pitch.
* `NGSpice_AutoChaos_No_Fingers` runs NGSpice with the included predictive
  technology models and fixed source/drain parasitics on each device.
* `NGSpice_AutoChaos_Fingers` is the NGSpice backend with finger-aware sizing
  and geometry-dependent parasitics.

The four are kept as separate frozen packages on purpose, so each one matches
exactly the configuration that produced its reported runs. The NGSpice
packages work with no licenses at all. The model files (45nm_bulk.pm,
45nm_HP.pm, ptm22lp.lib, ptm65.lib) are freely distributable PTM/BPTM cards
and ship in the repo. The Cadence packages need your own Spectre install and
PDK. No PDK content is included here.

## Setting up

```
conda create -n autochaos python=3.12
conda activate autochaos
pip install -r requirements.txt
```

Everything runs on CPU. Simulation dominates the runtime, so a GPU buys you nothing.

## Running the NGSpice packages

Install NGSpice (apt install ngspice on Debian/Ubuntu, or the Windows
installer). If ngspice is not on your PATH, put the full path in the training
config. The line is `ngspice_bin` in `autochaos/configs/training_config_ngspice.yaml`:

```yaml
ngspice_bin: "C:\\ngspice\\Spice64\\bin\\ngspice.exe"
```

Always validate before training. Validate mode skips Ray completely, builds
real netlists from the map config, runs real simulations for a couple of
random-action episodes, and prints per-corner metrics and rewards. It is the
fastest way to catch a wrong path, a broken template, or a sweep mismatch.

Linux/macOS, from the package folder:

```
PYTHONPATH=. python autochaos/train.py --config autochaos/configs/training_config_ngspice.yaml --mode validate --engine ngspice --episodes 1
```

Windows PowerShell:

```
$env:PYTHONPATH="."
python autochaos\train.py --config autochaos\configs\training_config_ngspice.yaml --mode validate --engine ngspice --episodes 1
```

You should see the init banner (template path, sweep, binary), then 551/551
slice completions per evaluation and a finite reward per step. Negative
rewards are normal here because random designs are rarely chaotic. Validate floors at
2 episodes, so expect around 12 evaluations.

Training:

```
PYTHONPATH=. python autochaos/train.py --config autochaos/configs/training_config_ngspice.yaml --mode train --engine ngspice --workers 4 --episodes 1000 2>&1 | tee run.log
```

The iteration count comes from `--episodes` on the command line, not from the
YAML. `--workers` is the number of Ray rollout workers. Each worker runs its
own parallel pool of ngspice processes (`max_workers` in the config), so total
process count is roughly workers times max_workers. Keep that product sensible
for your core count.

## Running the Cadence packages

Two edits before anything runs, both site-specific:

1. In `templates/netlist_3t.scs` (and any template you use), point the
   `include` line at your PDK's Spectre models. The `section={PROCESS}`
   placeholder stays, and the framework fills it per corner.
2. Set `cadence_setup` in the training config to your site's Cadence
   environment script. The default in the engine is the path from the
   machine this was developed on and will not exist at your site.

Validate first (the `-u` matters because without it Python block-buffers through the
pipe and the terminal stays silent for the whole first simulation):

```
PYTHONPATH=. python -u autochaos/train.py --config autochaos/configs/training_config_3t_parallel20.yaml --mode validate --engine cadence --episodes 1 2>&1 | tee validate_3t.log
```

Each evaluation is a full nested parametric DC sweep in one Spectre run, then
an OCEAN export to CSV, then the Python chaos analysis. Watch for
"Spectre finished (rc=0)", the PSF count line (551 DC sweeps expected for the
3T config), "OCEAN finished (rc=0)", and a metrics line per corner. On a
shared server your simulations queue for licenses behind other users. The
`lqtimeout` setting covers that wait, and a few minutes of silence per
simulation is normal.

Full training on a shared server:

```
export RAY_TMPDIR=$HOME/ray_tmp
export RAY_memory_monitor_refresh_ms=0
PYTHONPATH=. python -u autochaos/train.py --config autochaos/configs/training_config_3t_parallel20.yaml --mode train --engine cadence --workers 20 --episodes 1000 2>&1 | tee run.log
```

The two exports are not optional folklore. Ray puts its object store under
/tmp by default and will happily fill the root filesystem on a long run.
RAY_TMPDIR moves it to your home. The memory monitor gets disabled because on
a busy multi-user box it kills workers that are merely waiting on licenses.
Pick `--workers` against your license pool. Every worker holds a Spectre slot
while simulating (and an OCEAN slot while exporting), so 20 workers means up
to 20 concurrent licenses of each. If someone else is running too, size down.

If you Ctrl-C a Cadence run, check for orphaned simulator processes before
launching again. They keep holding licenses:

```
ps -u $USER -o pid,etime,cmd | grep spectre
```

and kill anything stale. Orphaned Spectres from crashed runs are the classic
way a shared license pool quietly starves.

## What a run writes

Everything lands under `runs/` in the package folder:

* `run_config.json` is written at launch and records the exact configuration
  the run started with. Treat this file, not the YAML in configs/, as the
  authoritative record of a run. YAMLs get edited between runs, but launch
  records do not.
* `progress.csv` has per-iteration training metrics.
* `metrics_cache.db` is a SQLite cache of every simulated design and corner,
  keyed by parameters, corner and model file. Repeat evaluations of the same
  point hit the cache instead of the simulator.
* a top-designs JSON with the best designs found so far, and checkpoints
  under `runs/checkpoints/`.

Between unrelated experiments, clear the old state:

```
rm -rf runs
```

or at least delete `runs/metrics_cache.db`. The cache key covers parameters,
corner and the model file name, but not the netlist template contents, so if
you edit the template itself, delete the cache or the run will keep returning
metrics for the old circuit.

## Using your own circuit

The engines contain no circuit. A circuit is a netlist template in
`templates/` plus a map config: parameter ranges, a nominal starting design,
the DC sweep block, the PVT corners, and the reward thresholds. The full
walkthrough, including how the sweep and the CSV format connect and what to
set when your output node is not net5, is in
`docs/ADDING_YOUR_OWN_CIRCUIT.md`. A map config that does not declare a
netlist template is a hard error. There is no built-in circuit to fall back
to.

## Things worth knowing

* CR and ALE are only meaningful within one device model. Numbers from
  gpdk045 and numbers from BPTM are not comparable. The same geometry can be
  strongly chaotic under one and nearly dead under the other.
* The NGSpice packages vary voltage and temperature per corner but use the
  single BPTM model card for all process corners. The engine prints a note
  about this at startup. The Cadence packages select real PDK process
  sections per corner.
* MSCMI configs and netlists in the Cadence packages take much longer per
  simulation than the 3T ones, and the timeouts in their training config
  reflect that. Start with 3T when checking a new setup.

## License

No license file yet. Until one is added, default copyright applies. If you
want to build on this, open an issue.
