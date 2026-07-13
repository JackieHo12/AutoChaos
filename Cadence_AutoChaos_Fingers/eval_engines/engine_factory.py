import json
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
    if name == "cadence":
        return _CadenceFullEngine(config)
    raise ValueError(
        f"Unknown engine '{engine_name}'. Choose from: "
        "'mock', 'python' (aliases: 'matlab', 'chaotic'), or 'cadence'."
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


class _CadenceFullEngine:

    def __init__(self, config: Dict):
        from eval_engines.cadence.cadence_engine import CadenceEngine
        from eval_engines.python.chaos_analyzer import PythonChaosAnalyzer
        project_dir = config.get(
            "project_dir",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        )
        cadence_config = {
            "project_dir": project_dir,
            "runs_base_dir": config.get("runs_base_dir", os.path.join(project_dir, "runs")),
            "templates_dir": config.get("templates_dir", os.path.join(project_dir, "templates")),
            "cadence_setup": config.get("cadence_setup", "/ece-tools/cadence/cadence-setup.rc"),
            "spectre_timeout": int(config.get("spectre_timeout", 180)),
            "ocean_timeout": int(config.get("ocean_timeout", 180)),
            "ocean_home_base": config.get("ocean_home_base", "/tmp/ocean_home_autochaos"),
            "map_config_path": config.get("map_config_path", "map_config.yaml"),
            "spectre_retries": config.get("spectre_retries", 1),
            "lqtimeout": config.get("lqtimeout",
                                    max(60, int(config.get("spectre_timeout", 180)) - 300)),
        }
        self.cadence = CadenceEngine(cadence_config)
        self.analyzer = PythonChaosAnalyzer()
        self._step = 0
        runs_dir = os.path.join(project_dir, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        self._db_path = os.path.join(runs_dir, "metrics_cache.db")
        self._cache_wait_s = int(config.get("cache_wait_s", 500))
        self._init_db()
        print("[CadenceFullEngine] Ready (SQLite WAL cache + Spectre + OCEAN + Python)")
    def _connect(self):
        con = sqlite3.connect(self._db_path, timeout=60.0)
        con.execute("PRAGMA busy_timeout=60000")  # must be FIRST: later pragmas inherit it
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con
    def _db_op(self, fn, what, attempts=5):
        delay = 0.5
        last = None
        for attempt in range(attempts):
            con = None
            try:
                con = self._connect()
                out = fn(con)
                con.commit()
                return out
            except sqlite3.OperationalError as e:
                last = e
                print(f"[CadenceFullEngine] DB busy during {what} "
                      f"(attempt {attempt+1}/{attempts}): {e}")
            finally:
                if con is not None:
                    try: con.close()
                    except Exception: pass
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)
        raise last
    def _init_db(self):
        self._db_op(
            lambda con: con.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, mle REAL, ale REAL, cr REAL, "
                "bif REAL, power REAL, area REAL, pending INTEGER DEFAULT 0)"
            ),
            "init_db", attempts=8)
    def _make_cache_key(self, params: Dict) -> str:
        def fmt(v):
            try: return f"{float(v):.12g}"
            except (TypeError, ValueError): return str(v)
        return "|".join(f"{k}={fmt(params[k])}" for k in sorted(params.keys()))
    def _row_to_metrics(self, row) -> Dict:
        return {
            'MLE': row[0], 'ALE': row[1], 'chaotic_ratio': row[2],
            'bifurcation_density': row[3], 'power_mw': row[4], 'area_um2': row[5],
        }
    def evaluate(self, params: Dict, run_id: str = None, **kwargs) -> Dict:
        self._step += 1
        cache_key = self._make_cache_key(params)
        i_own_claim = False
        for _ in range(max(1, self._cache_wait_s // 2)):
            def _read_or_claim(con):
                cur = con.execute(
                    "SELECT mle, ale, cr, bif, power, area, pending FROM cache WHERE key=?",
                    (cache_key,))
                row = cur.fetchone()
                if row is not None and row[6] == 0:
                    return ("hit", row)
                if row is None:
                    ins = con.execute(
                        "INSERT OR IGNORE INTO cache (key, pending) VALUES (?, 1)",
                        (cache_key,))
                    if ins.rowcount == 1:
                        return ("claimed", None)  # we own the pending row
                    return ("wait", None)  # lost the race - another worker owns it
                return ("wait", None)  # pending=1 owned by another worker
            try:
                state, row = self._db_op(_read_or_claim, "read/claim")
            except sqlite3.OperationalError:
                state, row = ("claimed_unverified", None)
            if state == "hit":
                metrics = self._row_to_metrics(row[:6])
                print(f"[CadenceFullEngine] Cache hit at step {self._step}: "
                      f"MLE={metrics['MLE']:.4f}, ALE={metrics['ALE']:.4f}, CR={metrics['chaotic_ratio']:.4f}")
                return dict(metrics)
            if state in ("claimed", "claimed_unverified"):
                i_own_claim = (state == "claimed")
                break
            print("[CadenceFullEngine] Waiting for duplicate sim to finish...")
            time.sleep(2)
        else:
            print(f"[CadenceFullEngine] Wait window ({self._cache_wait_s}s) expired - re-simulating")
        try:
            if run_id is None:
                remapped = self.cadence._remap_params(params)
                run_id = self.cadence._make_run_id(remapped)
            csv_path = self.cadence.evaluate(params, run_id=run_id)
            raw = self.analyzer.analyze(csv_path)
            metrics = {
                'MLE': raw.get('MLE', 0.0),
                'ALE': raw.get('ALE', 0.0),
                'chaotic_ratio': raw.get('chaotic_ratio', 0.0),
                'bifurcation_density': 0.0,
                'power_mw': 0.0,
                'area_um2': 0.0,
            }
        except Exception:
            if i_own_claim:
                try:
                    self._db_op(
                        lambda con: con.execute(
                            "DELETE FROM cache WHERE key=? AND pending=1", (cache_key,)),
                        "claim cleanup")
                except Exception as ce:
                    print(f"[CadenceFullEngine] WARNING: claim cleanup failed "
                          f"(orphan pending row will be swept at next launch): {ce}")
            raise
        try:
            self._db_op(
                lambda con: con.execute(
                    "INSERT OR REPLACE INTO cache (key, mle, ale, cr, bif, power, area, pending) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (cache_key, metrics['MLE'], metrics['ALE'], metrics['chaotic_ratio'],
                     metrics['bifurcation_density'], metrics['power_mw'], metrics['area_um2'])),
                "result write")
        except Exception as we:
            print(f"[CadenceFullEngine] WARNING: cache write failed after successful sim "
                  f"({we}) - returning live metrics anyway")
        try:
            run_dir = os.path.dirname(csv_path)
            if os.path.exists(run_dir) and os.path.isdir(run_dir):
                import shutil; shutil.rmtree(run_dir)
        except Exception as de:
            print(f"[CadenceFullEngine] WARNING: run dir cleanup failed: {de}")
        return dict(metrics)
    def close(self):
        self.cadence.close()
