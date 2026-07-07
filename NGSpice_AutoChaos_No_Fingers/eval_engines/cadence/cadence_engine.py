import hashlib
import os
import shutil
import subprocess
import time
import uuid
from typing import Dict, Optional

class SpectreSimulationError(RuntimeError):
    pass

NETLIST_TEMPLATE = """\

simulator lang=spectre

global 0

parameters Vc_val=0 vin=0 W_PM0={W_PM0} L_PM0={L_PM0} W_PM1={W_PM1} L_PM1={L_PM1} W_NM0={W_NM0} L_NM0={L_NM0}

include "/ece-tools/open_pdks/GPDK/gpdk045_v_6_0/gpdk045/../models/spectre/gpdk045.scs" section={PROCESS}

subckt Topology2 VIN Vc Vdd Vout gnd
    PM0 (Vout Vc Vdd Vdd) g45p1lvt w=W_PM0 l=L_PM0 nf=1 as=16.8f ad=16.8f ps=520n pd=520n nrd=1.16667 nrs=1.16667 sa=140n sb=140n sd=160n sca=226.00151 scb=0.11734 scc=0.02767 m=1
    PM1 (gnd VIN Vout Vdd) g45p1lvt w=W_PM1 l=L_PM1 nf=1 as=1.12p ad=1.12p ps=16.28u pd=16.28u nrd=17.5m nrs=17.5m sa=140n sb=140n sd=160n sca=44.95198 scb=0.05033 scc=0.00571 m=1
    NM0 (Vout VIN gnd gnd) g45n1lvt w=W_NM0 l=L_NM0 nf=1 as=35f ad=35f ps=780n pd=780n nrd=560m nrs=560m sa=140n sb=140n sd=160n sca=121.81260 scb=0.07142 scc=0.01388 m=1

ends Topology2

I0 (net3 net2 net4 net5 0) Topology2

V2 (net2 0) vsource dc=Vc_val type=dc

V1 (net3 0) vsource dc=vin type=dc

V0 (net4 0) vsource dc={VDD} type=dc

simulatorOptions options psfversion="1.1.0" reltol=1e-3 vabstol=1e-6 iabstol=1e-12 temp={TEMP} scalem=1.0 scale=1.0 compatible=spectre

vc_sweep sweep param=Vc_val start=0 stop=1.1 step=0.002 {{
    vin_sweep dc param=vin start=0 stop=1.1 step=0.002 write="spectre.dc" maxiters=150 maxsteps=10000 annotate=status

}}

saveOptions options save=allpub

"""

OCEAN_SCRIPT_TEMPLATE = """\

raw = getShellEnvVar("RAW_DIR")

out_csv = getShellEnvVar("OUT_CSV")

printf("RAW_DIR = %s\\n" raw)

printf("OUT_CSV = %s\\n" out_csv)

n_vc = 551

n_vin = 551

allData = makeTable("allData" 0.0)

for(vc_idx 0 550
    fname = strcat(raw "/vc_sweep-" sprintf(nil "%03d" vc_idx) "_vin_sweep.dc")
    openResults(fname)
    wave = getData("net5" ?result "dc")
    if(wave != nil then
        yvec = drGetWaveformYVec(wave)
        n = drVectorLength(yvec)
        for(j 0 (n-1)
            allData[vc_idx * n_vin + j] = drGetElem(yvec j)
        )
    else
        printf("WARNING: net5 not found in %s\\n" fname)
        for(j 0 (n_vin-1)
            allData[vc_idx * n_vin + j] = 0.0
        )
    )

)

openResults(strcat(raw "/vc_sweep-000_vin_sweep.dc"))

wave0 = getData("net5" ?result "dc")

xvec = drGetWaveformXVec(wave0)

port = outfile(out_csv "w")

for(i 0 (n_vin-1)
    vin_val = drGetElem(xvec i)
    fprintf(port "%g" vin_val)
    for(vc_idx 0 550
        vout_val = allData[vc_idx * n_vin + i]
        fprintf(port ",%g,0" vout_val)
    )
    fprintf(port "\\n")

)

close(port)

printf("CSV export complete: %s\\n" out_csv)

exit()

"""

class CadenceEngine:
    CADENCE_SETUP = "/ece-tools/cadence/cadence-setup.rc"
    def __init__(self, config: Dict):
        self.project_dir = os.path.abspath(config.get("project_dir", "."))
        self.runs_base_dir = config.get("runs_base_dir", os.path.join(self.project_dir, "runs"))
        self.templates_dir = config.get("templates_dir", os.path.join(self.project_dir, "templates"))
        self.cadence_setup = config.get("cadence_setup", self.CADENCE_SETUP)
        self.spectre_timeout = int(config.get("spectre_timeout", 180))
        self.ocean_timeout = int(config.get("ocean_timeout", 180))
        self.worker_tag = config.get("worker_tag", f"pid{os.getpid()}")
        ocean_home_base = config.get("ocean_home_base", "/tmp/ocean_home_autochaos")
        self.ocean_home = os.path.join(ocean_home_base, self.worker_tag)
        os.makedirs(self.runs_base_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.ocean_home, exist_ok=True)
        print("[CadenceEngine] Initialized")
        print(f"[CadenceEngine]   project_dir      : {self.project_dir}")
        print(f"[CadenceEngine]   runs_base        : {self.runs_base_dir}")
        print(f"[CadenceEngine]   ocean_home       : {self.ocean_home}")
        print(f"[CadenceEngine]   spectre_timeout  : {self.spectre_timeout}")
        print(f"[CadenceEngine]   ocean_timeout    : {self.ocean_timeout}")
    def evaluate(self, params: Dict, run_id: Optional[str] = None) -> str:
        params = self._remap_params(params)
        if run_id is None:
            run_id = self._make_run_id(params)
        run_dir = os.path.join(self.runs_base_dir, run_id)
        psf_dir = os.path.join(run_dir, "psf")
        os.makedirs(psf_dir, exist_ok=True)
        netlist_path = os.path.join(run_dir, "topology2_dc.scs")
        ocean_script_path = os.path.join(run_dir, "export_dc.ocn")
        csv_path = os.path.join(run_dir, "result.csv")
        print(f"[CadenceEngine] === Step {run_id} ===")
        print(f"[CadenceEngine] Params: {self._fmt_params(params)}")
        self._create_netlist(params, netlist_path)
        self._write_ocean_script(ocean_script_path)
        self._run_spectre(netlist_path, psf_dir)
        self._run_ocean(psf_dir, csv_path, ocean_script_path)
        if os.path.exists(psf_dir):
            shutil.rmtree(psf_dir)
            print("[CadenceEngine] PSF dir removed (disk saved)")
        print(f"[CadenceEngine] CSV ready: {csv_path}")
        return csv_path
    def _remap_params(self, params: Dict) -> Dict:
        remap = {
            "W_M1": "W_PM0", "L_M1": "L_PM0",
            "W_M2": "W_PM1", "L_M2": "L_PM1",
            "W_M3": "W_NM0", "L_M3": "L_NM0",
        }
        remapped = {remap.get(k, k): v for k, v in params.items()}
        defaults = {
            "W_PM0": 120e-9, "L_PM0": 45e-9,
            "W_PM1": 8e-6,   "L_PM1": 45e-9,
            "W_NM0": 250e-9, "L_NM0": 500e-9,
        }
        for k, v in defaults.items():
            if k not in remapped:
                print(f"[CadenceEngine] WARNING: {k} missing, using default {v}")
                remapped[k] = v
        return remapped
    def _create_netlist(self, params: Dict, netlist_path: str):
        def to_spectre(val: float) -> str:
            val = float(val)
            if val >= 1e-3:
                return f"{val:.6g}"
            if val >= 1e-6:
                return f"{val * 1e6:.6g}u"
            if val >= 1e-9:
                return f"{val * 1e9:.6g}n"
            if val >= 1e-12:
                return f"{val * 1e12:.6g}p"
            return f"{val:.6g}"
        content = NETLIST_TEMPLATE.format(
            W_PM0=to_spectre(params["W_PM0"]), L_PM0=to_spectre(params["L_PM0"]),
            W_PM1=to_spectre(params["W_PM1"]), L_PM1=to_spectre(params["L_PM1"]),
            W_NM0=to_spectre(params["W_NM0"]), L_NM0=to_spectre(params["L_NM0"]),
            VDD=params.get("VDD", 1.1),
            TEMP=params.get("TEMP", 27),
            PROCESS=params.get("PROCESS", "tt"),
        )
        with open(netlist_path, "w") as f:
            f.write(content)
        print(f"[CadenceEngine] Netlist written: {netlist_path}")
    def _write_ocean_script(self, ocean_script_path: str):
        with open(ocean_script_path, "w") as f:
            f.write(OCEAN_SCRIPT_TEMPLATE)
    def _run_spectre(self, netlist_path: str, psf_dir: str):
        inner_cmd = (
            f"source {self.cadence_setup} && "
            f"spectre -format psfbin -raw {psf_dir} {netlist_path}"
        )
        print(f"[CadenceEngine] Running Spectre... timeout={self.spectre_timeout}s")
        t0 = time.time()
        result = subprocess.run(
            ["bash", "-lc", inner_cmd],
            shell=False,
            capture_output=True,
            text=True,
            timeout=self.spectre_timeout,
            cwd=self.project_dir,
        )
        elapsed = time.time() - t0
        print(f"[CadenceEngine] Spectre finished in {elapsed:.1f}s (rc={result.returncode})")
        if result.returncode != 0:
            print(f"[CadenceEngine] Spectre stderr:\n{result.stderr[-2000:]}")
            raise SpectreSimulationError(f"Spectre failed (rc={result.returncode}). Check: {netlist_path}")
        psf_files = os.listdir(psf_dir)
        n_dc = sum(1 for f in psf_files if f.endswith("_vin_sweep.dc"))
        print(f"[CadenceEngine] PSF files: {len(psf_files)} total, {n_dc} DC sweeps")
        if n_dc < 551:
            raise SpectreSimulationError(f"Spectre produced only {n_dc}/551 DC sweep files.")
    def _run_ocean(self, psf_dir: str, csv_path: str, ocean_script_path: str):
        abs_psf_dir = os.path.abspath(psf_dir)
        abs_csv_path = os.path.abspath(csv_path)
        abs_ocean_script = os.path.abspath(ocean_script_path)
        inner_cmd = (
            f"source {self.cadence_setup} && "
            f"HOME={self.ocean_home} "
            f"RAW_DIR='{abs_psf_dir}' "
            f"OUT_CSV='{abs_csv_path}' "
            f"ocean -nograph -replay {abs_ocean_script}"
        )
        print(f"[CadenceEngine] Running OCEAN export... timeout={self.ocean_timeout}s")
        t0 = time.time()
        result = subprocess.run(
            ["bash", "-lc", inner_cmd],
            shell=False,
            capture_output=True,
            text=True,
            timeout=self.ocean_timeout,
            cwd=self.project_dir,
        )
        elapsed = time.time() - t0
        print(f"[CadenceEngine] OCEAN finished in {elapsed:.1f}s (rc={result.returncode})")
        if result.returncode != 0:
            print(f"[CadenceEngine] OCEAN stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"OCEAN failed (rc={result.returncode})")
        if not os.path.exists(abs_csv_path):
            print(f"[CadenceEngine] OCEAN stdout:\n{result.stdout[-3000:]}")
            raise RuntimeError(f"OCEAN ran but CSV not created: {abs_csv_path}")
        csv_size = os.path.getsize(abs_csv_path)
        print(f"[CadenceEngine] CSV created: {abs_csv_path} ({csv_size / 1024 / 1024:.1f} MB)")
        if csv_size < 1000:
            raise RuntimeError(f"CSV suspiciously small ({csv_size} bytes).")
    def _make_run_id(self, params: Dict) -> str:
        param_str = "_".join(f"{k}{v:.12g}" if not isinstance(v, str) else f"{k}{v}" for k, v in sorted(params.items()))
        h = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"run_{h}_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:6]}"
    def _fmt_params(self, params: Dict) -> str:
        parts = []
        for k, v in params.items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                parts.append(f"{k}={v}")
                continue
            if v >= 1e-3:
                parts.append(f"{k}={v:.3g}")
            elif v >= 1e-6:
                parts.append(f"{k}={v * 1e6:.3g}u")
            elif v >= 1e-9:
                parts.append(f"{k}={v * 1e9:.3g}n")
            else:
                parts.append(f"{k}={v:.3g}")
        return ", ".join(parts)
    def close(self):
        pass

