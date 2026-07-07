import os
import sqlite3
import time
from typing import Dict


def create_engine(engine_name: str, config: Dict):
    name = str(engine_name).lower().strip()
    if name == "mock":
        from eval_engines.utils.mock_engine import MockEngine
        return MockEngine(config)
    if name in ("matlab", "chaotic"):
        return _MatlabEngine(config)
    if name == "ngspice":
        return _NGSpiceFullEngine(config)
    raise ValueError(
        f"Unknown engine '{engine_name}'. Choose from: "
        "'mock', 'python' (aliases: 'matlab', 'chaotic'), or 'ngspice'."
    )


class _MatlabEngine:

    def __init__(self, config: Dict):
        from eval_engines.python.chaos_analyzer import PythonChaosAnalyzer
        self.analyzer = PythonChaosAnalyzer()
        self.default_csv_path = config.get("default_csv_path", "data/test_result.csv")
        print(f"[PythonEngine] Ready. Default CSV: {self.default_csv_path}")
    def evaluate(self, params: Dict, csv_path: str = None, **kwargs) -> Dict:
        path = csv_path or self.default_csv_path
        raw = self.analyzer.analyze(path)
        return {
            "MLE": raw.get("MLE", 0.0),
            "ALE": raw.get("ALE", 0.0),
            "chaotic_ratio": raw.get("chaotic_ratio", 0.0),
            "bifurcation_density": 0.0,
            "power_mw": 0.0,
            "area_um2": 0.0,
        }
    def close(self):
        pass


class _NGSpiceFullEngine:

    def __init__(self, config: Dict):
        from eval_engines.ngspice.ngspice_engine import NGSpiceEngine
        from eval_engines.python.chaos_analyzer import PythonChaosAnalyzer
        project_dir = config.get(
            "project_dir",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        )
        ngspice_config = {
            "project_dir":    project_dir,
            "runs_base_dir":  config.get("runs_base_dir",  os.path.join(project_dir, "runs")),
            "templates_dir":  config.get("templates_dir",  os.path.join(project_dir, "templates")),
            "ngspice_bin":    config.get("ngspice_bin", "ngspice"),
            "model_file":     config.get("model_file",     "45nm_bulk.pm"),
            "ngspice_timeout":int(config.get("ngspice_timeout", 30)),
            "max_workers":    int(config.get("max_workers", 24)),
            "topology":       config.get("topology", "3t"),
            "map_config_path":config.get("map_config_path", ""),
        }
        self.ngspice = NGSpiceEngine(ngspice_config)
        self.analyzer = PythonChaosAnalyzer(
            max_workers=int(config.get("max_workers", 8))
        )
        self._step = 0
        self._retries = int(config.get("ngspice_retries", 1))
        self._cache_wait_s = float(config.get("cache_wait_s", 60.0))
        runs_dir = os.path.join(project_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        self._db_path = os.path.join(runs_dir, "metrics_cache.db")
        self._init_db()
        print("[NGSpiceFullEngine] Ready")
        print(f"[NGSpiceFullEngine]   model_file : {ngspice_config['model_file']}")
        print(f"[NGSpiceFullEngine]   ngspice_bin: {self.ngspice.ngspice_bin}")
        print(f"[NGSpiceFullEngine]   topology   : {ngspice_config['topology']}")
        print(f"[NGSpiceFullEngine]   cache      : {self._db_path}")
        print(f"[NGSpiceFullEngine]   cache_wait : {self._cache_wait_s:.0f}s, retries={self._retries}")
    def _connect(self):
        con = sqlite3.connect(self._db_path, timeout=60.0)
        con.execute("PRAGMA busy_timeout=60000")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con
    def _db_op(self, fn, what, attempts=5):
        delay = 0.5
        last = None
        for attempt in range(attempts):
            con = None
            try:
                con = self._connect()  # connect INSIDE try: pragmas can lock too
                result = fn(con)
                con.commit()
                return result
            except sqlite3.OperationalError as e:
                last = e
                print(f"[NGSpiceFullEngine] DB lock on {what} "
                      f"(attempt {attempt+1}/{attempts}): {e}")
                time.sleep(delay)
                delay = min(delay * 2.0, 8.0)
            finally:
                if con is not None:
                    try: con.close()
                    except Exception: pass
        raise last
    def _init_db(self):
        def _create(con):
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, mle REAL, ale REAL, cr REAL, "
                "bif REAL, power REAL, area REAL, pending INTEGER DEFAULT 0)"
            )
        self._db_op(_create, "init_db", attempts=8)
    def _make_cache_key(self, params: Dict) -> str:
        def fmt(v):
            try: return f"{float(v):.12g}"
            except (TypeError, ValueError): return str(v)
        base = "|".join(f"{k}={fmt(params[k])}" for k in sorted(params.keys()))
        return base + "|MODEL=" + os.path.basename(str(self.ngspice.model_file))
    def _row_to_metrics(self, row) -> Dict:
        return {
            "MLE": row[0], "ALE": row[1], "chaotic_ratio": row[2],
            "bifurcation_density": row[3], "power_mw": row[4], "area_um2": row[5],
        }
    def evaluate(self, params: Dict, **kwargs) -> Dict:
        self._step += 1
        cache_key = self._make_cache_key(params)
        i_own_claim = False
        # claim/wait loop; window driven by cache_wait_s
        wait_iters = max(1, int(self._cache_wait_s / 2.0))
        for w in range(wait_iters + 1):
            def _check_and_claim(con):
                cur = con.execute(
                    "SELECT mle, ale, cr, bif, power, area, pending "
                    "FROM cache WHERE key=?", (cache_key,))
                row = cur.fetchone()
                if row is not None and row[6] == 0:
                    return ("hit", row[:6])
                if row is None:
                    ins = con.execute(
                        "INSERT OR IGNORE INTO cache (key, pending) VALUES (?, 1)",
                        (cache_key,))
                    if ins.rowcount == 1:  # we own the claim
                        return ("claimed", None)
                return ("wait", None)
            try:
                status, payload = self._db_op(_check_and_claim, "check_and_claim")
            except sqlite3.OperationalError:
                status, payload = ("claimed_unverified", None)
            if status == "hit":
                metrics = self._row_to_metrics(payload)
                print(f"[NGSpiceFullEngine] Cache hit step {self._step}: "
                      f"ALE={metrics['ALE']:.4f} CR={metrics['chaotic_ratio']:.4f}")
                return dict(metrics)
            if status in ("claimed", "claimed_unverified"):
                i_own_claim = (status == "claimed")
                break
            if w % 5 == 0:
                print(f"[NGSpiceFullEngine] Waiting for duplicate sim... ({w*2}s)")
            time.sleep(2)
        else:
            print("[NGSpiceFullEngine] Wait window expired - simulating without claim")
        try:
            # simulate with one retry on transient failure
            csv_path = None
            last_exc = None
            for attempt in range(self._retries + 1):
                try:
                    csv_path = self.ngspice.evaluate(params)
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < self._retries:
                        print(f"[NGSpiceFullEngine] Sim failed (attempt "
                              f"{attempt+1}/{self._retries+1}): {e} - retrying")
                        time.sleep(1.0)
            if csv_path is None:
                raise last_exc
            raw = self.analyzer.analyze(csv_path)
            metrics = {
                "MLE":                 raw.get("MLE",           0.0),
                "ALE":                 raw.get("ALE",           0.0),
                "chaotic_ratio":       raw.get("chaotic_ratio", 0.0),
                "bifurcation_density": 0.0,
                "power_mw":            0.0,
                "area_um2":            0.0,
            }
            print(f"[NGSpiceFullEngine] Step {self._step}: "
                  f"ALE={metrics['ALE']:.4f} CR={metrics['chaotic_ratio']:.4f}")
            # cache write FIRST (with retry), run-dir cleanup AFTER
            def _store(con):
                con.execute(
                    "INSERT OR REPLACE INTO cache "
                    "(key, mle, ale, cr, bif, power, area, pending) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (cache_key, metrics["MLE"], metrics["ALE"],
                     metrics["chaotic_ratio"], metrics["bifurcation_density"],
                     metrics["power_mw"], metrics["area_um2"]))
            try:
                self._db_op(_store, "store_result")
            except Exception as e:
                # A cache failure must never become a lost result
                print(f"[NGSpiceFullEngine] WARNING: cache write failed after "
                      f"retries ({e}) - returning live metrics anyway")
            import shutil as _shutil
            run_dir = os.path.dirname(csv_path)
            if os.path.isdir(run_dir):
                _shutil.rmtree(run_dir, ignore_errors=True)
                print(f"[NGSpiceEngine] Raw dir removed. CSV: {csv_path}")
            return dict(metrics)
        except Exception:
            # only delete the pending row we own
            if i_own_claim:
                def _release(con):
                    con.execute(
                        "DELETE FROM cache WHERE key=? AND pending=1",
                        (cache_key,))
                try:
                    self._db_op(_release, "release_claim")
                except Exception as e:
                    print(f"[NGSpiceFullEngine] WARNING: claim release failed: {e}")
            raise
    def close(self):
        try:
            self.ngspice.close()
        except Exception:
            pass
