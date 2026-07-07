
import hashlib
import os
import shutil
import subprocess
import time
import uuid
import yaml
from typing import Dict, Optional

class SpectreSimulationError(RuntimeError):
    pass


class CadenceEngine:
    CADENCE_SETUP = "/ece-tools/cadence/cadence-setup.rc"
    def __init__(self, config: Dict):
        self.project_dir = os.path.abspath(config.get("project_dir", "."))
        self.runs_base_dir = config.get("runs_base_dir", os.path.join(self.project_dir, "runs"))
        self.templates_dir = config.get("templates_dir", os.path.join(self.project_dir, "templates"))
        self.cadence_setup = config.get("cadence_setup", self.CADENCE_SETUP)
        self.spectre_timeout = int(config.get("spectre_timeout", 900))
        self.ocean_timeout = int(config.get("ocean_timeout", 900))
        self.lqtimeout = int(config.get("lqtimeout", max(60, self.spectre_timeout - 300)))
        self.spectre_retries = int(config.get("spectre_retries", 1))
        self.failures_dir = os.path.join(self.runs_base_dir, "failures")
        self.worker_tag = config.get("worker_tag", f"pid{os.getpid()}")
        self.map_config_path = os.path.abspath(config.get("map_config_path", "map_config.yaml"))
        ocean_home_base = config.get("ocean_home_base", "/tmp/ocean_home_autochaos")
        self.ocean_home = os.path.join(ocean_home_base, f"pid{os.getpid()}")
        os.makedirs(self.runs_base_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.ocean_home, exist_ok=True)
        try:
            with open(self.map_config_path) as f:
                map_cfg = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"[CadenceEngine] Cannot load map config {self.map_config_path}: {e}")


        self._nominal_defaults = {k: float(v) for k, v in map_cfg.get("nominal", {}).items()}


        netlist_tpl_path = map_cfg.get("netlist_template")
        if not netlist_tpl_path:
            raise RuntimeError(
                "[CadenceEngine] map config missing 'netlist_template' key. "
                "Add:  netlist_template: templates/your_circuit.scs"
            )
        if not os.path.isabs(netlist_tpl_path):
            netlist_tpl_path = os.path.join(self.project_dir, netlist_tpl_path)
        with open(netlist_tpl_path) as f:
            self._netlist_template = f.read()
        print(f"[CadenceEngine]   netlist_template : {netlist_tpl_path}")


        # sweep dimensions derived from dc_sweep in the map config
        dc = map_cfg.get("dc_sweep", {})
        vc_start = float(dc.get("Vc_start", 0.0))
        vc_stop  = float(dc.get("Vc_stop",  1.1))
        vc_step  = float(dc.get("Vc_step",  0.005))
        vn_start = float(dc.get("Vin_start", 0.0))
        vn_stop  = float(dc.get("Vin_stop",  1.1))
        vn_step  = float(dc.get("Vin_step",  0.001))
        self._output_node    = dc.get("output_node", "Xnp1")
        self._expected_n_vc = round((vc_stop - vc_start) / vc_step) + 1
        self._expected_n_vin = round((vn_stop - vn_start) / vn_step) + 1
        print(f"[CadenceEngine]   output_node      : {self._output_node}")
        print(f"[CadenceEngine]   expected n_vc    : {self._expected_n_vc}")
        print(f"[CadenceEngine]   expected n_vin   : {self._expected_n_vin}")


        _NL = chr(10)
        _ocean_lines = [
            'raw = getShellEnvVar("RAW_DIR")',
            'out_csv = getShellEnvVar("OUT_CSV")',
            'printf("RAW_DIR = %s' + _NL + '" raw)',
            'printf("OUT_CSV = %s' + _NL + '" out_csv)',
            'n_vc = ' + str(self._expected_n_vc),
            'n_vin = ' + str(self._expected_n_vin),
            'allData = makeTable("allData" 0.0)',
            'for(vc_idx 0 ' + str(self._expected_n_vc - 1),
            '    fname = strcat(raw "/vc_sweep-" sprintf(nil "%03d" vc_idx) "_vin_sweep.dc")',
            '    openResults(fname)',
            '    wave = getData("' + self._output_node + '" ?result "dc")',
            '    if(wave != nil then',
            '        yvec = drGetWaveformYVec(wave)',
            '        n = drVectorLength(yvec)',
            '        for(j 0 (n-1)',
            '            allData[vc_idx * n_vin + j] = drGetElem(yvec j)',
            '        )',
            '    else',
            '        printf("WARNING: ' + self._output_node + ' not found in %s' + _NL + '" fname)',
            '        for(j 0 (n_vin-1)',
            '            allData[vc_idx * n_vin + j] = 0.0',
            '        )',
            '    )',
            ')',
            'openResults(strcat(raw "/vc_sweep-000_vin_sweep.dc"))',
            'wave0 = getData("' + self._output_node + '" ?result "dc")',
            'xvec = drGetWaveformXVec(wave0)',
            'port = outfile(out_csv "w")',
            'for(i 0 (n_vin-1)',
            '    vin_val = drGetElem(xvec i)',
            '    fprintf(port "%g" vin_val)',
            '    for(vc_idx 0 ' + str(self._expected_n_vc - 1),
            '        vout_val = allData[vc_idx * n_vin + i]',
            '        fprintf(port ",%g" vout_val)',
            '    )',
            '    fprintf(port "\\n" "")',
            ')',
            'close(port)',
        ]
        ocean_script = _NL.join(_ocean_lines) + _NL
        self._ocean_script_path = os.path.join(self.runs_base_dir, f"export_dc_{os.getpid()}.ocn")
        with open(self._ocean_script_path, "w") as f:
            f.write(ocean_script)
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
        circuit_name = os.path.splitext(os.path.basename(self.map_config_path))[0].replace("map_config_", "")
        netlist_path = os.path.join(run_dir, f"{circuit_name}_dc.scs")
        csv_path = os.path.join(run_dir, "result.csv")
        print(f"[CadenceEngine] === Step {run_id} ===")
        print(f"[CadenceEngine] Params: {self._fmt_params(params)}")
        self._create_netlist(params, netlist_path)
        self._run_spectre(netlist_path, psf_dir)
        self._run_ocean(psf_dir, csv_path)
        if os.path.exists(psf_dir):
            shutil.rmtree(psf_dir)
            print("[CadenceEngine] PSF dir removed (disk saved)")
        print(f"[CadenceEngine] CSV ready: {csv_path}")
        return csv_path
    def _remap_params(self, params: Dict) -> Dict:
        remapped = dict(params)
        for k, v in self._nominal_defaults.items():
            if k not in remapped:
                print(f"[CadenceEngine] WARNING: {k} missing, using nominal default {v}")
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
        # Dynamic: all tunable params from map config
        fmt = {k: to_spectre(v) for k, v in params.items()
               if k not in ("VDD", "TEMP", "PROCESS")}
        fmt["VDD"]     = params.get("VDD", 1.1)
        fmt["TEMP"]    = params.get("TEMP", 27)
        fmt["PROCESS"] = params.get("PROCESS", "tt")
        content = self._netlist_template.format(**fmt)
        with open(netlist_path, "w") as f:
            f.write(content)
        print(f"[CadenceEngine] Netlist written: {netlist_path}")
    def _run_spectre(self, netlist_path: str, psf_dir: str):
        inner_cmd = (
            f"source {self.cadence_setup} && "
            f"spectre +lqtimeout {self.lqtimeout} "
            f"-ahdllibdir '{os.path.dirname(netlist_path)}/ahdlcmi' "
            f"-format psfbin -raw '{psf_dir}' '{netlist_path}'"
        )
        attempts = 1 + max(0, self.spectre_retries)
        for attempt in range(attempts):
            print(f"[CadenceEngine] Running Spectre... timeout={self.spectre_timeout}s"
                  + (f" (retry {attempt}/{attempts-1})" if attempt else ""))
            t0 = time.time()
            try:
                result = subprocess.run(
                    ["bash", "-lc", inner_cmd],
                    shell=False,
                    capture_output=True,
                    text=True,
                    timeout=self.spectre_timeout,
                    cwd=self.project_dir,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                print(f"[CadenceEngine] Spectre TIMEOUT after {elapsed:.1f}s — raising SpectreSimulationError")
                raise SpectreSimulationError(f"Spectre timed out after {self.spectre_timeout}s")
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"[CadenceEngine] Spectre subprocess error after {elapsed:.1f}s: {exc}")
                raise SpectreSimulationError(f"Spectre subprocess failed: {exc}")
            elapsed = time.time() - t0
            print(f"[CadenceEngine] Spectre finished in {elapsed:.1f}s (rc={result.returncode})")
            if result.returncode == 0:
                psf_files = os.listdir(psf_dir)
                n_dc = sum(1 for f in psf_files if f.endswith("_vin_sweep.dc"))
                print(f"[CadenceEngine] PSF files: {len(psf_files)} total, {n_dc} DC sweeps")
                if n_dc < self._expected_n_vc:
                    run_dir = os.path.dirname(netlist_path)
                    self._preserve_failure(netlist_path, result,
                                           f"partial PSF: {n_dc}/{self._expected_n_vc}")
                    shutil.rmtree(run_dir, ignore_errors=True)
                    raise SpectreSimulationError(
                        f"Spectre produced only {n_dc}/{self._expected_n_vc} DC sweep files.")
                return
            # rc != 0: spectre's real reason is on STDOUT, not stderr
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            stdout_tail = result.stdout[-3000:] if result.stdout else ""
            print(f"[CadenceEngine] Spectre stderr:\n{stderr_tail}")
            print(f"[CadenceEngine] Spectre stdout (failure reason usually here):\n{stdout_tail}")
            if attempt < attempts - 1:
                time.sleep(5)
                continue
            # Final attempt failed: preserve evidence, then clean up
            self._preserve_failure(netlist_path, result, f"rc={result.returncode}")
            run_dir = os.path.dirname(netlist_path)
            shutil.rmtree(run_dir, ignore_errors=True)
            raise SpectreSimulationError(
                f"Spectre failed (rc={result.returncode}) after {attempts} attempt(s). "
                f"Evidence in {self.failures_dir}")
    def _preserve_failure(self, netlist_path: str, result, reason: str):
        try:
            os.makedirs(self.failures_dir, exist_ok=True)
            existing = os.listdir(self.failures_dir)
            if len(existing) >= 200:
                return
            tag = f"{time.strftime('%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:4]}"
            dst = os.path.join(self.failures_dir, tag)
            os.makedirs(dst, exist_ok=True)
            if os.path.exists(netlist_path):
                shutil.copy2(netlist_path, dst)
            with open(os.path.join(dst, "spectre_output.txt"), "w") as f:
                f.write(f"REASON: {reason}\n\n=== STDOUT ===\n")
                f.write(result.stdout or "")
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr or "")
        except Exception as e:
            print(f"[CadenceEngine] WARNING: could not preserve failure evidence: {e}")
    def _run_ocean(self, psf_dir: str, csv_path: str):
        abs_psf_dir = os.path.abspath(psf_dir)
        abs_csv_path = os.path.abspath(csv_path)
        inner_cmd = (
            f"source {self.cadence_setup} && "
            f"HOME={self.ocean_home} "
            f"RAW_DIR='{abs_psf_dir}' "
            f"OUT_CSV='{abs_csv_path}' "
            f"ocean -nograph -replay {self._ocean_script_path}"
        )
        print(f"[CadenceEngine] Running OCEAN export... timeout={self.ocean_timeout}s")
        t0 = time.time()
        try:
            result = subprocess.run(
                ["bash", "-lc", inner_cmd],
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.ocean_timeout,
                cwd=self.project_dir,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"OCEAN timed out after {self.ocean_timeout}s")
        except Exception as exc:
            raise RuntimeError(f"OCEAN subprocess failed: {exc}")
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
        try:
            if os.path.exists(self._ocean_script_path):
                os.remove(self._ocean_script_path)
        except Exception:
            pass
