# Adding Your Own Circuit

This guide walks through everything needed to train AutoChaos on a new
chaotic circuit. No framework code changes are required: a circuit is defined
by (1) a netlist template, (2) a map configuration, and (3) a training
configuration. The three-transistor files in each package are the working
examples to copy from.

## 1. The netlist template

Create `templates/netlist_<yourcircuit>.<ext>` (`.cir` for NGSpice, `.scs`
for Spectre). Three kinds of placeholder are substituted per evaluation:

- **Sized dimensions** - one `{NAME}` per tunable parameter, e.g.
  `W={W_PM0} L={L_NM0}`. The names must match the `params:` keys in your map
  config exactly.
- **Corner variables** - `{PROCESS}`, `{VDD}`, `{TEMP}` wherever the process
  section, supply, and temperature enter your netlist.
- **Fingered packages only** - `NF={NF_<NAME>}` for each fingered device; the
  finger count is derived from the width and the configured `finger_step`,
  never searched directly.

Device models:

- **NGSpice** - include the model file shipped at the package root (or your
  own PTM card): the training config's `model_file:` key selects it.
- **Cadence** - the template's `include` line must point at *your* PDK's
  Spectre models, e.g.
  `include "/your/pdk/path/models/spectre/gpdk045.scs" section={PROCESS}`.
  The repository does not ship any PDK content.

Non-fingered NGSpice templates carry fixed source/drain parasitics
(`AS=/AD=/PS=/PD=`) on each device; fingered templates omit them and let the
model compute geometry-dependent parasitics from W, L, and NF. Follow the
same convention for a new circuit so the fingered/non-fingered comparison
means the same thing it does for the included circuits.

## 2. The map configuration

Copy `autochaos/configs/map_config_3t.yaml` and edit:

```yaml
netlist_template: templates/netlist_<yourcircuit>.cir

params:              # one entry per tunable dimension: [min, max, step]
  W_M1: [120e-9, 10e-6, 120e-9]
  L_M1: [45e-9, 2e-6, 25e-9]
  # ...

nominal:             # a hand-sized starting design if there is one (injected into the warm start)
  W_M1: 1.0e-6
  L_M1: 100e-9

dc_sweep:            # the transfer-characteristic sweep the chaos analysis uses
  Vc_start: 0.0
  Vc_stop: 1.1       # match your VDD
  Vc_step: 0.002
  Vin_start: 0.0
  Vin_stop: 1.1
  Vin_step: 0.002
  output_node: <your output node name>
  VDD: 1.1
  temperature: 27

pvt_corners:         # the corner set; names must match your model sections
- process: tt
  VDD: 1.1
  temp: 27
- process: ss
  VDD: 1.045
  temp: 70
- process: ff
  VDD: 1.155
  temp: 0

pvt_reward:
  tau_CR: 0.35       # see section 4
  tau_ALE: 0.20
  w_CR: 0.5
  w_ALE: 0.5
  alpha: 0.6         # worst-corner vs mean-corner mixing in the robustness reward
  bonus: 2.0         # success bonus for all-pass designs
  penalty: 1.0
  progressive_tau: 0.25   # nominal-CR gate below which SS/FF are skipped
  robust_credit: 1.0
```

Key semantics to know:

- `params` step sizes define the discrete search grid.
- A design is **all-pass** only if it meets both `tau_CR` and `tau_ALE` at
  every corner; all-pass designs receive `bonus` on top of the robustness
  reward.
- `progressive_tau` is the gate: a design whose nominal-corner CR is below it
  is not simulated at the other corners, which saves large amounts of
  simulation time. It is distinct from `tau_CR`.
- Fingered packages additionally take `finger_mode: true` and
  `finger_step: <pitch>` (e.g. `120e-9`); the finger count of each device is
  `max(1, width // finger_step)` computed exactly.

## 3. The training configuration

Copy `autochaos/configs/training_config_ngspice.yaml` (or the Cadence
equivalent) and edit the environment block: the map config path, the
simulator binary or launch settings, `model_file` (NGSpice), timeouts, and
the worker count. PPO hyperparameters (learning rate, batch, network) are
circuit-independent defaults that transfer well; change them only with
reason. The number of training iterations is given on the command line with
`--episodes`, not in the YAML.

Timeout guidance: set the per-simulation timeout comfortably above your
circuit's worst observed simulation time on your machine - larger circuits
need more (the included examples use 20 minutes for a 3-transistor circuit
and 40 for a 23-parameter circuit on a heavily shared server).

## 4. Choosing the reward emphasis for a new circuit

Simulate your hand-sized `nominal` design once (the validate flow prints its
metrics) and read its chaotic ratio:

- **Baseline CR well below 1.0** - the chaotic ratio is the binding metric.
  Weight the metrics evenly (`w_CR: 0.5, w_ALE: 0.5`) or toward CR, and set
  `tau_CR` somewhat above the baseline CR so the threshold is a stretch
  target the search can reach.
- **Baseline CR at or near 1.0 (saturated)** - CR carries no gradient; the
  average Lyapunov exponent is the binding metric. Weight ALE
  (`w_ALE: 0.6, w_CR: 0.4` worked well), set `tau_CR` near 1.0 so only
  fully chaotic designs pass, and set `tau_ALE` above the baseline ALE.

Set `progressive_tau` low enough that promising designs reach the corner
evaluation but high enough to skip clearly dead ones; a small fraction of
your `tau_CR` is a reasonable start.

## 5. The DC sweep, the CSV, and the chaos analyzer

The sweep geometry is declared in the `dc_sweep` block of the map config,
and neither backend requires code changes when it changes - but the two
backends consume it differently:

- **NGSpice**: the engine builds its sweep vectors directly from `dc_sweep`
  (`Vc_start/stop/step`, `Vin_start/stop/step`), and the template's
  `.dc VIN {VIN_START:.6g} {VIN_STOP:.6g} {VIN_STEP:.6g}` line receives the
  same values, so editing the config is the whole change. Parameter
  substitution is circuit-agnostic: every key of the design point is offered
  to the template, and in the fingered package every width `W_<name>`
  additionally provides a derived `NF_<name>` finger count, so a new circuit
  needs only matching placeholder names, exactly as on the Cadence backend.
  If the simulated point count nevertheless disagrees with `dc_sweep`, the
  engine prints a loud resampling warning naming both numbers.
- **Cadence**: the sweep is executed by the netlist template's own analysis
  statements (the `vc_sweep sweep param=... start/stop/step` and nested
  `vin_sweep dc param=...` lines), so a new sweep is set in the template, and
  `dc_sweep` in the map config must be kept in sync with it. The engine uses
  `dc_sweep` to size the OCEAN export and to validate the result; on a
  mismatch it fails loudly with "Spectre produced only X/Y DC sweep files"
  rather than producing wrong metrics.

That works because the chaos analyzer is sweep-agnostic. The engines write
the simulated transfer characteristics to a CSV with one row per Vin point:
column 0 is the Vin value, followed by one `(Vout, 0)` column pair per Vc
point (the zero is a legacy filler from the original MATLAB analyzer's
expected layout). The analyzer infers the number of Vin rows and Vc columns
from the file itself and scales its perturbation from the actual Vin spacing,
so a 551x551 sweep and a 221x1101 sweep flow through the same code. Two
consequences worth knowing:

- The Vc axis *values* never enter the analysis - the chaotic ratio is the
  fraction of Vc columns whose Lyapunov exponent is positive, so only the
  column count matters.
- If you replace an engine or exporter with one that writes a different CSV
  layout (for example plain Vout columns without the filler), the single
  function to adapt is `load_csv` in
  `eval_engines/python/chaos_analyzer.py`.

Two names must stay in sync on the NGSpice backend: the `output_node` key of
`dc_sweep` (which the engine uses to locate the signal in the raw file) and
the `.save V(<node>)` line of your netlist template. If your circuit's output
node is not `net5`, set both.

The analyzer's own knobs (`TRUNCATE`, `SEQ_LEN`, `INIT_STATE`,
`DELTA_SCALE` at the top of `chaos_analyzer.py`) control the iterated-map
computation, not the sweep; they are circuit-independent analysis settings
and normally should not change.

## 6. Validate, then train

```
PYTHONPATH=. python autochaos/train.py --config autochaos/configs/<your_training>.yaml \
    --mode validate --engine <ngspice|cadence>
```

Validate mode bypasses Ray entirely, builds one netlist from your `nominal`
design, runs the simulator once per corner, and runs the chaos analysis - it
will surface template placeholder mistakes, model-path problems, and sweep
misconfiguration in minutes. Only then launch `--mode train`.

Every run writes `run_config.json` next to its outputs. Treat that file - not
the YAML in the configs folder - as the record of what a run actually used:
YAML files get edited between runs, launch records do not.
