import hashlib
import os
import re
import shutil
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
import numpy as np


class NGSpiceSimulationError(RuntimeError):
    pass


def _fmt(val: float) -> str:
    val = float(val)
    if val == 0.0:
        return "0"
    a = abs(val)
    if a >= 1e-3: return f"{val:.6g}"
    if a >= 1e-6: return f"{val*1e6:.6g}u"
    if a >= 1e-9: return f"{val*1e9:.6g}n"
    if a >= 1e-12: return f"{val*1e12:.6g}p"
    return f"{val:.6g}"


def parse_ngspice_ascii_raw(raw_path: str, signal_name: str, n_expected: int) -> np.ndarray:
    with open(raw_path, "r", errors="replace") as f:
        text = f.read()
    values_pos = text.find("Values:")
    if values_pos == -1:
        raise ValueError(f"No 'Values:' in {raw_path}")
    header = text[:values_pos]
    values_text = text[values_pos + len("Values:"):]
    n_vars, n_points, var_names = 0, 0, []
    in_variables = False
    for line in header.splitlines():
        s = line.strip()
        m = re.match(r"No\.\s*Variables:\s*(\d+)", s, re.IGNORECASE)
        if m: n_vars = int(m.group(1))
        m = re.match(r"No\.\s*Points:\s*(\d+)", s, re.IGNORECASE)
        if m: n_points = int(m.group(1))
        if re.match(r"^Variables\s*:", s, re.IGNORECASE):
            in_variables = True; continue
        if in_variables:
            m = re.match(r"\s*\d+\s+(\S+)\s+\S+", line)
            if m: var_names.append(m.group(1).lower())
    if n_vars == 0 or n_points == 0:
        raise ValueError(f"Bad header in {raw_path}: n_vars={n_vars} n_points={n_points}")
    target = signal_name.lower().replace(" ", "")
    col = None
    for i, name in enumerate(var_names):
        if name.replace(" ","") == target:
            col = i; break
    if col is None:
        tb = target.replace("(","").replace(")","")
        for i, name in enumerate(var_names):
            if name.replace("(","").replace(")","").replace(" ","") == tb:
                col = i; break
    if col is None:
        node = target.replace("v(","").replace(")","")
        for i, name in enumerate(var_names):
            if node in name: col = i; break
    if col is None:
        col = min(1, n_vars - 1)
    all_nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", values_text)
    all_nums = [float(x) for x in all_nums]
    stride = n_vars + 1
    needed = stride * n_points
    if len(all_nums) < needed:
        raise ValueError(f"Not enough numbers: {len(all_nums)} < {needed}")
    vout = np.array([all_nums[i * stride + 1 + col] for i in range(n_points)], dtype=np.float64)
    if len(vout) != n_expected:
        print(f"[NGSpiceEngine] WARNING: raw has {len(vout)} Vin points, "
              f"expected {n_expected}; resampling. Check dc_sweep vs the "
              f"netlist template's .dc line.")
        vout = np.interp(np.linspace(0,1,n_expected), np.linspace(0,1,len(vout)), vout)
    return vout


class NGSpiceEngine:
    NGSPICE_DEFAULT = "ngspice"
    VC_START = 0.0; VC_STOP = 1.1; VC_STEP = 0.002
    VIN_START = 0.0; VIN_STOP = 1.1; VIN_STEP = 0.002
    def __init__(self, config: Dict):
        self.project_dir = os.path.abspath(config.get("project_dir", "."))
        self.runs_base_dir = config.get("runs_base_dir", os.path.join(self.project_dir, "runs"))
        self.model_file = os.path.abspath(config.get("model_file", os.path.join(self.project_dir, "45nm_bulk.pm")))
        self.ngspice_bin = config.get("ngspice_bin", self.NGSPICE_DEFAULT)
        self.timeout = int(config.get("ngspice_timeout", 30))
        self.worker_tag = config.get("worker_tag", f"pid{os.getpid()}")
        self.max_workers = int(config.get("max_workers", 24))
        self.map_config_path = config.get("map_config_path", os.path.join(self.project_dir, "autochaos", "configs", "map_config_3t.yaml"))
        import yaml as _yaml
        with open(self.map_config_path) as _f:
            _mc = _yaml.safe_load(_f)
        _dc = (_mc or {}).get("dc_sweep", {})
        self.VC_START = float(_dc.get("Vc_start", self.VC_START))
        self.VC_STOP = float(_dc.get("Vc_stop", self.VC_STOP))
        self.VC_STEP = float(_dc.get("Vc_step", self.VC_STEP))
        self.VIN_START = float(_dc.get("Vin_start", self.VIN_START))
        self.VIN_STOP = float(_dc.get("Vin_stop", self.VIN_STOP))
        self.VIN_STEP = float(_dc.get("Vin_step", self.VIN_STEP))
        self.output_signal = "v(%s)" % _dc.get("output_node", "net5")
        _tpl_rel = (_mc or {}).get("netlist_template")
        if not _tpl_rel:
            raise ValueError(f"{self.map_config_path} must declare netlist_template")
        _tpl_path = _tpl_rel if os.path.isabs(_tpl_rel) else os.path.join(self.project_dir, _tpl_rel)
        if not os.path.isfile(_tpl_path):
            raise FileNotFoundError(
                f"netlist_template declared in {self.map_config_path} "
                f"but not found: {_tpl_path}")
        with open(_tpl_path, encoding="utf-8") as _tf:
            self._netlist_template = _tf.read()
        print(f"[NGSpiceEngine]   netlist_template : {_tpl_path}")
        os.makedirs(self.runs_base_dir, exist_ok=True)
        import shutil as _shutil
        if not (os.path.isfile(self.ngspice_bin) or _shutil.which(self.ngspice_bin)):
            # conventional Windows installer location as a last resort
            _win_default = r"C:\ngspice\Spice64\bin\ngspice.exe"
            if os.name == "nt" and os.path.isfile(_win_default):
                print(f"[NGSpiceEngine]   '{self.ngspice_bin}' not on PATH; using {_win_default}")
                self.ngspice_bin = _win_default
            else:
                raise FileNotFoundError(
                    f"[NGSpiceEngine] ngspice not found: {self.ngspice_bin} "
                    f"(not a file and not on PATH; set 'ngspice_bin' in the config)")
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"[NGSpiceEngine] Model file not found: {self.model_file}")
        print("[NGSpiceEngine] Initialized")
        print(f"[NGSpiceEngine]   ngspice_bin  : {self.ngspice_bin}")
        print(f"[NGSpiceEngine]   model_file   : {self.model_file}")
        print(f"[NGSpiceEngine]   runs_base    : {self.runs_base_dir}")
        print(f"[NGSpiceEngine]   max_workers  : {self.max_workers} parallel NGSpice procs")
        print(f"[NGSpiceEngine]   timeout/run  : {self.timeout}s")
        print("[NGSpiceEngine]   NOTE: PROCESS corner param (tt/ss/ff) is NOT applied -\n"
              "[NGSpiceEngine]   the single BPTM model file is used for all corners, so\n"
              "[NGSpiceEngine]   corners vary VDD and TEMP only (VT corners, not full PVT).\n"
              "[NGSpiceEngine]   Document this in the thesis; see corner_model_files TODO.")
    def evaluate(self, params: Dict, run_id: Optional[str] = None) -> str:
        params = self._remap_params(params)
        if run_id is None:
            run_id = self._make_run_id(params)
        run_dir = os.path.join(self.runs_base_dir, run_id)
        raw_dir = os.path.join(run_dir, "raw")
        csv_path = os.path.join(run_dir, "result.csv")
        os.makedirs(raw_dir, exist_ok=True)
        print(f"[NGSpiceEngine] === {run_id} ===")
        print(f"[NGSpiceEngine] Params: {self._fmt_params(params)}")
        vc_values = np.arange(self.VC_START, self.VC_STOP + self.VC_STEP / 2, self.VC_STEP)
        vin_values = np.arange(self.VIN_START, self.VIN_STOP + self.VIN_STEP / 2, self.VIN_STEP)
        n_total = len(vc_values)
        Vout_matrix = np.zeros((len(vin_values), n_total), dtype=np.float64)
        # Write all netlists up front (fast)
        for idx, vc in enumerate(vc_values):
            self._write_netlist(params, vc, idx, os.path.join(raw_dir, f"run_{idx:03d}.cir"))
        def _run_one(idx):
            npath = os.path.join(raw_dir, f"run_{idx:03d}.cir")
            rpath = os.path.join(raw_dir, f"run_{idx:03d}.raw")
            vec, err = self._run_ngspice(npath, rpath, vin_values)
            return idx, vec, err
        t0 = time.time()
        completed = 0
        failures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_run_one, idx): idx for idx in range(n_total)}
            for future in as_completed(futures):
                idx, vout_vec, err = future.result()
                if err is not None:
                    failures.append((idx, err))
                else:
                    Vout_matrix[:, idx] = vout_vec
                completed += 1
                if completed % 50 == 0 or completed == n_total:
                    print(f"[NGSpiceEngine]   {completed}/{n_total} done ({time.time()-t0:.0f}s)")
        if failures:
            failures.sort(key=lambda x: x[0])
            print(f"[NGSpiceEngine] {len(failures)}/{n_total} slices FAILED - "
                  f"first: slice {failures[0][0]}: {failures[0][1][:200]}")
            self._preserve_failure(run_id, failures, raw_dir)
            shutil.rmtree(raw_dir, ignore_errors=True)
            raise NGSpiceSimulationError(
                f"{len(failures)}/{n_total} Vc slices failed "
                f"(first: {failures[0][1][:200]}). Evidence in runs/failures/{run_id}")
        elapsed = time.time() - t0
        print(f"[NGSpiceEngine] All {n_total} runs done in {elapsed:.1f}s ({self.max_workers} workers)")
        self._write_csv(vin_values, Vout_matrix, csv_path)
        shutil.rmtree(raw_dir)
        print(f"[NGSpiceEngine] Raw dir removed. CSV: {csv_path}")
        return csv_path
    def _remap_params(self, params: Dict) -> Dict:
        remapped = dict(params)
        import yaml
        try:
            with open(self.map_config_path) as f:
                map_cfg = yaml.safe_load(f)
            nominal_defaults = {k: float(v) for k, v in map_cfg.get("nominal", {}).items()}
        except Exception:
            nominal_defaults = {}
        for k, v in nominal_defaults.items():
            if k not in remapped:
                print(f"[NGSpiceEngine] WARNING: {k} missing, using nominal default {v}")
                remapped[k] = v
        return remapped
    def _write_netlist(self, params: Dict, vc_fixed: float, run_idx: int, netlist_path: str):
        # every design key is offered to the template
        subs = {
            "MODEL_FILE": self.model_file.replace("\\", "/"),
            "VC_FIXED": vc_fixed, "RUN_IDX": run_idx,
            "VDD": params.get("VDD", 1.1), "TEMP": params.get("TEMP", 27),
            "VIN_START": self.VIN_START, "VIN_STOP": self.VIN_STOP,
            "VIN_STEP": self.VIN_STEP,
        }
        subs["PROCESS"] = params.get("PROCESS", "tt")
        for k, v in params.items():
            if k in ("VDD", "TEMP", "PROCESS"):
                continue
            subs[k] = _fmt(v)
        content = self._netlist_template.format(**subs)
        with open(netlist_path, "w") as f:
            f.write(content)
    def _run_ngspice(self, netlist_path, raw_path, vin_values):
        try:
            run_kw = {}
            run_cwd = os.path.dirname(os.path.abspath(raw_path))
            run_kw["cwd"] = run_cwd
            raw_name = os.path.basename(raw_path)
            net_name = os.path.basename(netlist_path)
            if os.name != "nt":
                _si_path = os.path.join(run_cwd, ".spiceinit")
                if not os.path.exists(_si_path):
                    with open(_si_path, "w") as _sf:
                        _sf.write("set filetype=ascii\n")
            else:
                # hide per-slice console windows; stdout stays piped
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = subprocess.SW_HIDE
                run_kw["startupinfo"] = si
                run_kw["creationflags"] = 0x08000000
            result = subprocess.run(
                [self.ngspice_bin, "-b", "-r", raw_name, net_name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=self.timeout,
                **run_kw,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout after {self.timeout}s"
        except Exception as e:
            return None, f"subprocess error: {e}"
        if result.returncode != 0:
            tail = (result.stderr or "")[-500:]
            return None, f"rc={result.returncode}: {tail}"
        if not os.path.isfile(raw_path):
            tail = (result.stderr or "")[-500:]
            return None, f"no rawfile produced: {tail}"
        try:
            return parse_ngspice_ascii_raw(raw_path, self.output_signal, len(vin_values)), None
        except Exception as e:
            return None, f"parse failed: {e}"
    def _preserve_failure(self, run_id, failures, raw_dir):
        try:
            fdir = os.path.join(self.runs_base_dir, "failures")
            os.makedirs(fdir, exist_ok=True)
            if len(os.listdir(fdir)) >= 200:
                return
            dst = os.path.join(fdir, run_id)
            os.makedirs(dst, exist_ok=True)
            first_idx = failures[0][0]
            npath = os.path.join(raw_dir, f"run_{first_idx:03d}.cir")
            if os.path.isfile(npath):
                shutil.copy2(npath, dst)
            with open(os.path.join(dst, "errors.txt"), "w") as f:
                for idx, err in failures[:20]:
                    f.write(f"slice {idx}: {err}\n")
        except Exception as e:
            print(f"[NGSpiceEngine] WARNING: could not preserve failure evidence: {e}")
    def _write_csv(self, vin_values: np.ndarray, Vout_matrix: np.ndarray, csv_path: str):
        n_vin, n_vc = Vout_matrix.shape
        with open(csv_path, "w") as f:
            for i in range(n_vin):
                row = [f"{vin_values[i]:.8g}"]
                for j in range(n_vc):
                    row.append(f"{Vout_matrix[i,j]:.8g}")
                    row.append("0")
                f.write(",".join(row) + "\n")
        size_mb = os.path.getsize(csv_path) / 1024 / 1024
        print(f"[NGSpiceEngine] CSV written: {csv_path} ({size_mb:.1f} MB)")
        if size_mb < 0.1:
            raise NGSpiceSimulationError(f"CSV too small ({size_mb:.2f} MB)")
    def _make_run_id(self, params: Dict) -> str:
        s = "_".join(f"{k}{v:.12g}" if not isinstance(v,str) else f"{k}{v}" for k,v in sorted(params.items()))
        h = hashlib.md5(s.encode()).hexdigest()[:8]
        return f"run_{h}_{os.getpid()}_{time.time_ns()}_{uuid.uuid4().hex[:6]}"
    def _fmt_params(self, params: Dict) -> str:
        parts = []
        for k, v in params.items():
            try: v = float(v)
            except: parts.append(f"{k}={v}"); continue
            if v >= 1e-3: parts.append(f"{k}={v:.3g}")
            elif v >= 1e-6: parts.append(f"{k}={v*1e6:.3g}u")
            elif v >= 1e-9: parts.append(f"{k}={v*1e9:.3g}n")
            else: parts.append(f"{k}={v:.3g}")
        return ", ".join(parts)
    def close(self):
        pass

