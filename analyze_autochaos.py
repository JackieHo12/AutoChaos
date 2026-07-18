"""
AutoChaos run analyzer.

Files needed (same folder as script or under --run_dir):
  progress.csv        from ~/ray_results/PPO_chaos-map-v0_<timestamp>/
  result.json         from ~/ray_results/PPO_chaos-map-v0_<timestamp>/
  metrics_cache.db    SQLite cache from runs/ (preferred)
  metrics_cache.json  JSON cache from runs/ (legacy, also supported)
  lhs_pool.json       from runs/lhs_pool.json (optional)

Usage:
  python analyze_autochaos.py
  python analyze_autochaos.py --run_dir . --cache metrics_cache.db --lhs lhs_pool.json
  python analyze_autochaos.py --save

The --cache flag auto-detects format by extension (.db = SQLite, .json = JSON).
If omitted, it searches for metrics_cache.db first, then metrics_cache.json.
"""

import argparse, json, os, sys, glob, sqlite3, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
try:
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import pandas as pd
try:
    from scipy.stats import gaussian_kde
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    gaussian_kde = None

plt.rcParams.update({
    "figure.max_open_warning": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "axes.titlecolor": "#111111",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "grid.linestyle": "--",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#111111",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "legend.framealpha": 0.9,
    "figure.titlesize": 13,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

C_BLUE = "#1f77b4"
C_ORANGE = "#d65f00"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_PURPLE = "#9467bd"
C_BROWN = "#8c564b"
C_GREY = "#7f7f7f"
C_TEAL = "#17becf"

TOPOLOGY = "3-Transistor"
TAU_CR = 0.35
TAU_ALE = 0.20

W_ALE_R = 0.5
W_CR_R = 0.5
ALPHA_R = 0.6
ROBUST_CREDIT_R = 1.0

def _score_metric(value, tau):
    """Per-metric bounded score, mirrors reward_pvt.py _score()."""
    import math as _m
    v = max(float(value), 0.0)
    if tau <= 0.0: tau = 1e-9
    if v < tau:
        return 0.5 * (v / tau)
    e = (v - tau) / tau
    return 0.5 + 0.3 * _m.tanh(e) + 0.2 * min(e, 1.0)

def compute_design_reward(crs, ales,
                          tau_cr=None, tau_ale=None,
                          w_ale=None, w_cr=None, alpha=None):
    """Reward for a finished design from its per-corner CR/ALE lists.
    Mirrors reward_pvt.py compute_pvt_reward. Ranking key for the tables."""
    tau_cr = TAU_CR if tau_cr is None else tau_cr
    tau_ale = TAU_ALE if tau_ale is None else tau_ale
    w_ale = W_ALE_R if w_ale is None else w_ale
    w_cr = W_CR_R if w_cr is None else w_cr
    alpha = ALPHA_R if alpha is None else alpha
    if not crs or not ales:
        return -1.0
    n = len(crs)
    ale_wc = alpha * min(ales) + (1.0 - alpha) * (sum(ales) / n)
    cr_wc = alpha * min(crs) + (1.0 - alpha) * (sum(crs) / n)
    r_robust = w_ale * _score_metric(ale_wc, tau_ale) + w_cr * _score_metric(cr_wc, tau_cr)
    all_pass = all(c >= tau_cr for c in crs) and all(a >= tau_ale for a in ales)
    bonus = 2.0 if all_pass else 0.0
    if all_pass:
        return r_robust + bonus

    nom_cr = crs[0]
    progress = min(nom_cr / tau_cr, 1.0) if tau_cr > 0 else 1.0
    return -1.0 + progress + ROBUST_CREDIT_R * r_robust
KL_TARGET = 0.02
GRAD_CLIP = 0.5
PROG_GATE = 0.25


def find_latest_run():
    pattern = os.path.expanduser("~/ray_results/PPO_chaos-map-v0_*/progress.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return str(Path(files[-1]).parent) if files else None

def load_progress(run_dir):
    p = os.path.join(run_dir, "progress.csv")
    if not os.path.exists(p):
        print(f"ERROR: progress.csv not found in {run_dir}"); sys.exit(1)
    return pd.read_csv(p)

def load_results(run_dir):
    p = os.path.join(run_dir, "result.json")
    if not os.path.exists(p): return []
    results = []
    with open(p) as f:
        for line in f:
            try: results.append(json.loads(line.strip()))
            except: pass
    return results

def load_cache(path):
    """Load metrics cache from SQLite (.db) or JSON (.json)."""
    if not path or not os.path.exists(path):
        return {}

    if path.endswith(".db"):
        return _load_cache_sqlite(path)
    else:
        with open(path) as f:
            return json.load(f)

def _load_cache_sqlite(db_path):
    """Read SQLite metrics_cache.db into the same dict format as JSON."""
    cache = {}
    try:
        con = sqlite3.connect(db_path, timeout=10)
        rows = con.execute(
            "SELECT key, mle, ale, cr, bif, power, area, pending FROM cache"
        ).fetchall()
        con.close()
        for row in rows:
            key, mle, ale, cr, bif, power, area, pending = row
            if pending == 1:
                continue
            cache[key] = {
                "MLE": mle or 0.0,
                "ALE": ale or 0.0,
                "chaotic_ratio": cr or 0.0,
                "bifurcation_density": bif or 0.0,
                "power_mw": power or 0.0,
                "area_um2": area or 0.0,
            }
        print(f"  [SQLite] Loaded {len(cache)} entries from {db_path}"
              f" ({sum(1 for r in rows if r[7]==1)} pending skipped)")
    except Exception as e:
        print(f"  [SQLite] ERROR reading {db_path}: {e}")
    return cache

def _find_cache_file(directory="."):
    """Prefer .db over .json."""
    db_path = os.path.join(directory, "metrics_cache.db")
    json_path = os.path.join(directory, "metrics_cache.json")
    if os.path.exists(db_path):
        return db_path
    if os.path.exists(json_path):
        return json_path
    return None

def load_lhs(path):
    if not path or not os.path.exists(path): return []
    with open(path) as f: return json.load(f)

def parse_cache(cache):
    """Returns all_sims list and by_widths dict."""
    all_sims = []
    by_widths = defaultdict(dict)
    for k, v in cache.items():
        if not isinstance(v, dict): continue
        parts = dict(p.split("=") for p in k.split("|") if "=" in p)
        process = parts.pop("PROCESS", "tt")
        temp = float(parts.pop("TEMP", 27.0))
        vdd = float(parts.pop("VDD", 1.1))
        wk = "|".join(f"{pk}={pv}" for pk, pv in sorted(parts.items()))
        cr = float(v.get("chaotic_ratio", 0.0))
        ale = float(v.get("ALE", 0.0))
        all_sims.append({"wk": wk, "process": process,
                         "temp": temp, "vdd": vdd,
                         "cr": cr, "ale": ale, "idx": len(all_sims)})
        by_widths[wk][process] = {"cr": cr, "ale": ale}
    return all_sims, dict(by_widths)

def export_top_designs_json(designs, prefix, top_n=10):
    import json as _json
    out = []
    for i, d in enumerate(designs[:top_n]):
        params = {}
        for pair in d["widths"].split("|"):
            if "=" in pair:
                k, v = pair.split("=")
                params[k] = float(v)
        out.append({
            "rank": i + 1,
            "design_id": f"D{i+1}",
            "reward": d.get("reward"),
            "min_cr": d.get("min_cr"),
            "min_ale": d.get("min_ale"),
            "params": params,
        })
    path = f"{prefix}_top_designs.json"
    with open(path, "w") as f:
        _json.dump(out, f, indent=2)
    print(f"Exported top {len(out)} designs to {path}")
    return path

def get_pvt_designs(by_widths):
    """Return designs evaluated at all 3 corners, ranked by reward."""
    designs = []
    for wk, corners in by_widths.items():
        if "tt" not in corners or "ss" not in corners or "ff" not in corners:
            continue
        crs = [corners[p]["cr"] for p in ["tt","ss","ff"]]
        ales = [corners[p]["ale"] for p in ["tt","ss","ff"]]
        designs.append({
            "wk": wk,
            "widths": wk,
            "corners": corners,
            "tt": corners["tt"],
            "ss": corners["ss"],
            "ff": corners["ff"],
            "tt_cr": crs[0], "ss_cr": crs[1], "ff_cr": crs[2],
            "tt_ale": ales[0], "ss_ale": ales[1], "ff_ale": ales[2],
            "nom_cr": crs[0], "mean_cr": np.mean(crs), "min_cr": min(crs),
            "nom_ale": ales[0], "mean_ale": np.mean(ales), "min_ale": min(ales),
            "reward": compute_design_reward(crs, ales),
            "n_corners": 3,
        })

    def _rank_key(x):
        return (x["reward"], x["min_ale"], x["min_cr"], x["mean_ale"])
    designs.sort(key=_rank_key, reverse=True)
    return designs

def smooth(v, w=5):
    out = []
    for i in range(len(v)):
        lo = max(0, i-w//2); hi = min(len(v), i+w//2+1)
        out.append(np.mean(v[lo:hi]))
    return np.array(out)

def running_best(vals):
    best, out = float("-inf"), []
    for v in vals: best = max(best, v); out.append(best)
    return np.array(out)

def parse_params(wk):
    """Parse width key into nm values."""
    result = {}
    for p in wk.split("|"):
        k, v = p.split("=")
        result[k] = float(v) * 1e9
    return result

def fmt_params(wk, per_line=6, max_lines=None):
    """Compact one-or-multi-line parameter string. PROCESS/VDD/TEMP stripped."""
    try:
        items = []
        for p in wk.split("|"):
            if "=" not in p: continue
            k, v = p.split("=")
            if any(c in k for c in ("PROCESS", "VDD", "TEMP")): continue
            short = k.replace("W_", "W").replace("L_", "L")
            val = float(v) * 1e9
            items.append(f"{short}={val:.0f}n" if val < 1000 else f"{short}={val/1000:.2f}u")
        lines = [" ".join(items[i:i+per_line]) for i in range(0, len(items), per_line)]
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines] + ["..."]
        return "\n".join(lines)
    except Exception:
        return wk[:90]

def fig_reward_curve(df, save, prefix, topology_name="3-Transistor"):
    """Reward learning curve: raw, rolling mean, best-so-far vs environment steps."""
    col_mean = "env_runners/episode_return_mean"
    col_max = "env_runners/episode_return_max"
    col_steps = "num_env_steps_sampled_lifetime"

    iters = df["training_iteration"].values
    steps = df[col_steps].values if col_steps in df.columns else iters * 100
    r_mean = df[col_mean].values if col_mean in df.columns else np.full(len(df), np.nan)
    r_max = df[col_max].values if col_max in df.columns else np.full(len(df), np.nan)
    r_best = running_best(np.where(np.isnan(r_mean), -np.inf, r_mean))
    r_sm = smooth(r_mean, w=max(3, len(r_mean)//10))
    valid = r_mean[~np.isnan(r_mean)]
    trend = np.polyfit(np.arange(len(valid)), valid, 1)[0] if len(valid) >= 4 else 0

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, r_mean, color=C_GREY, linewidth=0.8, alpha=0.6,
            label="Mean episode reward")
    ax.plot(steps, r_sm, color=C_BLUE, linewidth=2.0,
            label=f"Smoothed mean (trend: {trend:+.4f}/iter)")
    ax.plot(steps, r_best, color=C_ORANGE, linewidth=1.8, linestyle="--",
            label="Best-so-far reward")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_xlabel("Environment Steps", fontsize=10)
    ax.set_ylabel("Episode Reward", fontsize=10)
    ax.set_title("AutoChaos Reward Progression During Training\n"
                 f"({topology_name} Chaotic Circuit)")
    ax.legend(loc="lower right")
    ax.annotate(f"Final mean:\n{r_mean[-1]:.3f}",
                xy=(steps[-1], r_mean[-1]),
                xytext=(steps[-1]*0.82, r_mean[-1]+0.08),
                fontsize=8, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.8))
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig1_reward_curve.png")
        print("  Saved: fig1_reward_curve")
    return fig

def fig_search_saturation(all_sims, save, prefix, topology_name="3-Transistor"):
    """Best-so-far and rolling nominal CR/ALE over TT evaluations in cache order."""
    tt_sims = [s for s in all_sims if s["process"] == "tt"]
    crs = [s["cr"] for s in tt_sims]
    ales = [s["ale"] for s in tt_sims]
    x = np.arange(1, len(crs)+1)

    best_cr = running_best(crs)
    best_ale = running_best(ales)
    roll_cr = smooth(crs, w=max(5, len(crs)//20))
    roll_ale = smooth(ales, w=max(5, len(ales)//20))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Search Saturation of Best Chaotic Metrics\n"
                 f"({TOPOLOGY} Circuit)",
                 fontsize=12)

    ax1.plot(x, crs, color=C_GREY, linewidth=0.5, alpha=0.4, label="Nominal CR (each eval)")
    ax1.plot(x, roll_cr, color=C_BLUE, linewidth=1.5, label="Rolling mean CR")
    ax1.plot(x, best_cr, color=C_ORANGE, linewidth=2.0, linestyle="--",
             label="Best nominal CR so far")
    ax1.axhline(TAU_CR, color=C_GREEN, linewidth=1.0, linestyle="-.",
                label=f"Target τ_CR = {TAU_CR}")
    ax1.set_ylabel("Chaotic Ratio (CR)", fontsize=10)
    ax1.set_title("Search Saturation of Best Chaotic Ratio")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=8)
    ax1.set_ylim(bottom=0)

    ax2.plot(x, ales, color=C_GREY, linewidth=0.5, alpha=0.4, label="Nominal ALE (each eval)")
    ax2.plot(x, roll_ale, color=C_BLUE, linewidth=1.5, label="Rolling mean ALE")
    ax2.plot(x, best_ale, color=C_ORANGE, linewidth=2.0, linestyle="--",
             label="Best nominal ALE so far")
    ax2.axhline(TAU_ALE, color=C_GREEN, linewidth=1.0, linestyle="-.",
                label=f"Target τ_ALE = {TAU_ALE}")
    ax2.set_xlabel("Evaluation Number (TT Corner Simulations)", fontsize=10)
    ax2.set_ylabel("Average Lyapunov Exponent (ALE)", fontsize=10)
    ax2.set_title("Search Saturation of Best Lyapunov Exponent")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=8)
    ax2.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    if save:
        fig.savefig(f"{prefix}_fig2_search_saturation.png")
        print("  Saved: fig2_search_saturation")
    return fig

def fig_pvt_funnel(all_sims, by_widths, save, prefix):
    """Bar chart of design counts surviving each screening stage."""
    tt_crs = [s["cr"] for s in all_sims if s["process"] == "tt"]
    total = len(tt_crs)

    stages = [
        ("Total TT\nevaluations", total,
         C_GREY),
        (f"CR > 0.10\n(weak chaos)", sum(1 for c in tt_crs if c > 0.10),
         C_TEAL),
        (f"CR > {PROG_GATE}\n(gate threshold)", sum(1 for c in tt_crs if c > PROG_GATE),
         C_BLUE),
        (f"CR > {TAU_CR}\n(τ_CR target)", sum(1 for c in tt_crs if c > TAU_CR),
         C_GREEN),
        ("Full PVT\nevaluated", sum(1 for w in by_widths.values()
                                               if "ss" in w or "ff" in w),
         C_ORANGE),
        ("All 3 corners\nevaluated", sum(1 for w in by_widths.values()
                                               if "tt" in w and "ss" in w and "ff" in w),
         C_PURPLE),
        (f"Min CR > {TAU_CR}\nacross PVT", sum(1 for w in by_widths.values()
                                               if "tt" in w and "ss" in w and "ff" in w and
                                               min(w[p]["cr"] for p in ["tt","ss","ff"]) > TAU_CR),
         C_RED),
    ]

    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    y = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(y, counts, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.65)

    for bar, count, tot in zip(bars, counts, [total]*len(counts)):
        pct = 100 * count / total if total > 0 else 0
        ax.text(bar.get_width() + total*0.005, bar.get_y() + bar.get_height()/2,
                f"{count:,}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Designs", fontsize=10)
    ax.set_title("Progressive PVT Screening of Candidate Chaotic Designs\n"
                 f"({TOPOLOGY} Circuit)")
    ax.set_xlim(0, total * 1.25)
    ax.invert_yaxis()
    ax.axvline(total, color=C_GREY, linewidth=0.5, linestyle=":")
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig3_pvt_funnel.png")
        print("  Saved: fig3_pvt_funnel")
    return fig

def fig_nominal_vs_worst(pvt_designs, save, prefix):
    """Scatter of nominal CR against worst-corner CR for three-corner designs."""
    if not pvt_designs:
        print("  SKIP fig4: no 3-corner designs"); return None

    nom = np.array([d["nom_cr"] for d in pvt_designs])
    worst = np.array([d["min_cr"] for d in pvt_designs])
    ale = np.array([d["mean_ale"] for d in pvt_designs])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sc = ax.scatter(nom, worst, c=ale, cmap="viridis", s=60,
                    edgecolors="black", linewidths=0.4, zorder=3, alpha=0.85)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Mean ALE across PVT", fontsize=9)

    lim = max(nom.max(), worst.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=C_GREY, linewidth=1.0,
            linestyle="--", label="y = x  (perfect robustness)", zorder=2)
    ax.axhline(TAU_CR, color=C_GREEN, linewidth=1.0, linestyle="-.",
               label=f"τ_CR = {TAU_CR}", zorder=2)
    ax.axvline(TAU_CR, color=C_GREEN, linewidth=1.0, linestyle="-.", zorder=2)
    for d in pvt_designs[:3]:
        ax.annotate(f"  #{pvt_designs.index(d)+1}",
                    xy=(d["nom_cr"], d["min_cr"]),
                    fontsize=8, color="#333333")

    ax.set_xlabel("Nominal CR (TT corner)", fontsize=10)
    ax.set_ylabel("Minimum CR (worst PVT corner)", fontsize=10)
    ax.set_title("Nominal vs. Worst-Case PVT Chaotic Ratio\n"
                 "(Each point = one fully PVT-evaluated design)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig4_nominal_vs_worst.png")
        print("  Saved: fig4_nominal_vs_worst")
    return fig

def fig_cr_ale_tradeoff(pvt_designs, save, prefix):
    """Scatter of mean CR against mean ALE, colored by worst-corner CR."""
    if not pvt_designs:
        print("  SKIP fig5: no 3-corner designs"); return None

    mean_cr = np.array([d["mean_cr"] for d in pvt_designs])
    mean_ale = np.array([d["mean_ale"] for d in pvt_designs])
    min_cr = np.array([d["min_cr"] for d in pvt_designs])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(mean_cr, mean_ale, c=min_cr, cmap="RdYlGn",
                    s=80, edgecolors="black", linewidths=0.4,
                    zorder=3, alpha=0.9, vmin=0, vmax=max(min_cr)*1.1)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Min CR across PVT (robustness)", fontsize=9)

    ax.axhline(TAU_ALE, color=C_BLUE, linewidth=1.0, linestyle="-.",
               label=f"τ_ALE = {TAU_ALE}", zorder=2)
    ax.axvline(TAU_CR, color=C_GREEN, linewidth=1.0, linestyle="-.",
               label=f"τ_CR = {TAU_CR}", zorder=2)
    for i, d in enumerate(pvt_designs[:3]):
        ax.annotate(f"  #{i+1}", xy=(d["mean_cr"], d["mean_ale"]),
                    fontsize=8, color="#333333")

    ax.set_xlabel("Mean CR across PVT Corners", fontsize=10)
    ax.set_ylabel("Mean ALE across PVT Corners", fontsize=10)
    ax.set_title("Tradeoff Between Chaotic Ratio and Lyapunov Exponent\n"
                 "(Color = worst-corner CR; top-right = best)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig5_cr_ale_tradeoff.png")
        print("  Saved: fig5_cr_ale_tradeoff")
    return fig

def fig_distribution_shift(all_sims, lhs_pool, save, prefix):
    """CR and ALE histograms, LHS initial pool against all RL-explored designs."""
    tt_sims = [s for s in all_sims if s["process"] == "tt"]

    if lhs_pool and isinstance(lhs_pool, list) and len(lhs_pool) > 0:
        lhs_crs = [e.get("CR_nominal", e.get("cr", 0)) for e in lhs_pool]
        lhs_ales = [e.get("ALE_nominal", e.get("ale", 0)) for e in lhs_pool]
    else:

        lhs_crs = [s["cr"] for s in tt_sims[:50]]
        lhs_ales = [s["ale"] for s in tt_sims[:50]]

    rl_crs = [s["cr"] for s in tt_sims]
    rl_ales = [s["ale"] for s in tt_sims]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Distribution Shift: Initial Sampling vs. RL-Explored Designs\n"
                 f"({TOPOLOGY} Circuit)", fontsize=12)

    bins_cr = np.linspace(0, max(max(lhs_crs,default=0), max(rl_crs,default=0))*1.05, 30)
    bins_ale = np.linspace(0, max(max(lhs_ales,default=0), max(rl_ales,default=0))*1.05, 30)

    ax = axes[0]
    ax.hist(rl_crs, bins=bins_cr, color=C_BLUE, alpha=0.65,
            label=f"RL-explored (n={len(rl_crs):,})", density=True, edgecolor="white")
    ax.hist(lhs_crs, bins=bins_cr, color=C_ORANGE, alpha=0.75,
            label=f"LHS initial (n={len(lhs_crs)})", density=True, edgecolor="white")
    ax.axvline(TAU_CR, color=C_GREEN, linewidth=1.5, linestyle="-.",
               label=f"τ_CR = {TAU_CR}")
    ax.set_xlabel("Chaotic Ratio (CR)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("CR Distribution")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(rl_ales, bins=bins_ale, color=C_BLUE, alpha=0.65,
            label=f"RL-explored (n={len(rl_ales):,})", density=True, edgecolor="white")
    ax.hist(lhs_ales, bins=bins_ale, color=C_ORANGE, alpha=0.75,
            label=f"LHS initial (n={len(lhs_ales)})", density=True, edgecolor="white")
    ax.axvline(TAU_ALE, color=C_GREEN, linewidth=1.5, linestyle="-.",
               label=f"τ_ALE = {TAU_ALE}")
    ax.set_xlabel("Average Lyapunov Exponent (ALE)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("ALE Distribution")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig6_distribution_shift.png")
        print("  Saved: fig6_distribution_shift")
    return fig

def fig_parameter_distribution(all_sims, by_widths, save, prefix, top_n=50):
    """Box plots of each parameter, all designs against the top-N by nominal CR."""
    tt_sims = [s for s in all_sims if s["process"] == "tt" and s["cr"] > PROG_GATE]
    tt_sims.sort(key=lambda x: x["cr"], reverse=True)
    top_sims = tt_sims[:top_n]

    if not top_sims:
        print("  SKIP fig7: no designs above progressive gate"); return None

    def _parse(wk):
        return dict(p.split("=") for p in wk.split("|") if "=" in p and not any(
            c in p for c in ("PROCESS", "VDD", "TEMP")))
    all_param_names = set()
    for s in top_sims:
        all_param_names.update(_parse(s["wk"]).keys())
    w_names = sorted([n for n in all_param_names if n.startswith("W_")])
    l_names = sorted([n for n in all_param_names if n.startswith("L_")])
    param_names = w_names + l_names
    if not param_names:
        print("  SKIP fig7: no parameters parsed"); return None
    n_p = len(param_names)

    param_data = defaultdict(list)
    for s in top_sims:
        parts = _parse(s["wk"])
        for pn in param_names:
            if pn in parts: param_data[pn].append(float(parts[pn]) * 1e9)
    all_param_data = defaultdict(list)
    for s in all_sims:
        if s["process"] != "tt": continue
        parts = _parse(s["wk"])
        for pn in param_names:
            if pn in parts: all_param_data[pn].append(float(parts[pn]) * 1e9)

    ncol = min(6, n_p) if n_p > 4 else n_p
    nrow = (n_p + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0*ncol + 1, 3.0*nrow + 0.5))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Parameter Distribution Among Top {top_n} Chaotic Designs\n"
                 f"(Ranked by nominal CR > {PROG_GATE}, {TOPOLOGY})",
                 fontsize=12, y=0.99)

    for idx, pn in enumerate(param_names):
        ax = axes[idx]
        data_top = param_data.get(pn, [])
        data_all = all_param_data.get(pn, [])
        if not data_top: ax.set_visible(False); continue
        ax.boxplot([data_all, data_top],
                   tick_labels=["All", f"Top{top_n}"],
                   patch_artist=True,
                   medianprops=dict(color=C_ORANGE, linewidth=2),
                   boxprops=dict(facecolor="lightsteelblue", alpha=0.7),
                   whiskerprops=dict(color=C_GREY), capprops=dict(color=C_GREY),
                   flierprops=dict(marker="o", markersize=2,
                                   markerfacecolor=C_GREY, alpha=0.3))
        ax.set_title(pn.replace("W_","W ").replace("L_","L "), fontsize=8.5)
        if idx % ncol == 0: ax.set_ylabel("nm", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(n_p, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        fig.savefig(f"{prefix}_fig7_parameter_distribution.png")
        print("  Saved: fig7_parameter_distribution")
    return fig

def fig_parameter_mode(pvt_designs, save, prefix, top_n=20):
    """Mode of each parameter across the top-N designs. Bar = most common value,
    whisker = full [min, max], color = fraction of designs sitting at the mode."""
    designs = pvt_designs[:top_n]
    if not designs:
        print("  SKIP fig_mode: no designs"); return None

    def _parse(wk):
        return dict(p.split("=") for p in wk.split("|") if "=" in p and not any(
            c in p for c in ("PROCESS", "VDD", "TEMP")))

    names = set()
    for d in designs:
        names.update(_parse(d["widths"]).keys())
    w_names = sorted([n for n in names if n.startswith("W_")])
    l_names = sorted([n for n in names if n.startswith("L_")])
    param_names = w_names + l_names
    if not param_names:
        print("  SKIP fig_mode: no params parsed"); return None

    from collections import Counter
    modes, mins, maxs, consistency = [], [], [], []
    for pn in param_names:
        vals = []
        for d in designs:
            parts = _parse(d["widths"])
            if pn in parts:
                try: vals.append(float(parts[pn]))
                except Exception: pass
        if not vals:
            modes.append(0.0); mins.append(0.0); maxs.append(0.0); consistency.append(0.0); continue

        nm_vals = [round(v * 1e9, 1) for v in vals]
        c = Counter(nm_vals)
        mode_nm, mode_count = c.most_common(1)[0]
        modes.append(mode_nm)
        mins.append(min(nm_vals)); maxs.append(max(nm_vals))
        consistency.append(mode_count / len(nm_vals))

    n_p = len(param_names)
    fig, ax = plt.subplots(figsize=(max(10, n_p * 0.55), 6))
    xs = np.arange(n_p)

    colors = [plt.cm.RdYlGn(c) for c in consistency]
    ax.bar(xs, modes, color=colors, edgecolor="#333", linewidth=0.6, zorder=3)

    for i in range(n_p):
        ax.plot([xs[i], xs[i]], [mins[i], maxs[i]], color="#444", linewidth=1.2, zorder=4)
        ax.plot([xs[i]-0.15, xs[i]+0.15], [maxs[i], maxs[i]], color="#444", linewidth=1.2, zorder=4)
        ax.plot([xs[i]-0.15, xs[i]+0.15], [mins[i], mins[i]], color="#444", linewidth=1.2, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(param_names, rotation=90, fontsize=8)
    ax.set_ylabel("Parameter value (nm)", fontsize=10)
    ax.set_title(f"Most Common Parameter Value (Mode) Across Top {len(designs)} Designs\n"
                 "Green bars = RL consistently homes to one size (a learned design rule); "
                 "red bars = variable transistors  |  whisker = full explored [min, max]",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label("Consistency (fraction of designs at the mode)", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig_parameter_mode.png", dpi=140)
        print("  Saved: fig_parameter_mode")
    return fig

def fig_parameter_mode_table(pvt_designs, save, prefix, top_n=20):
    designs = pvt_designs[:top_n]
    if not designs:
        print("  SKIP fig_mode_table: no designs"); return None

    def _parse(wk):
        return dict(p.split("=") for p in wk.split("|") if "=" in p and not any(
            c in p for c in ("PROCESS", "VDD", "TEMP")))

    names = set()
    for d in designs:
        names.update(_parse(d["widths"]).keys())
    w_names = sorted([n for n in names if n.startswith("W_")])
    l_names = sorted([n for n in names if n.startswith("L_")])
    param_names = w_names + l_names
    if not param_names:
        print("  SKIP fig_mode_table: no params parsed"); return None

    from collections import Counter
    entries = []
    for pn in param_names:
        vals = []
        for d in designs:
            parts = _parse(d["widths"])
            if pn in parts:
                try: vals.append(float(parts[pn]))
                except Exception: pass
        if not vals:
            continue
        nm_vals = [round(v * 1e9, 1) for v in vals]
        c = Counter(nm_vals)
        mode_nm, mode_count = c.most_common(1)[0]
        n = len(nm_vals)
        cons = mode_count / n
        def fmt_nm(x):
            return f"{x:.0f}n" if x < 1000 else f"{x/1000:.2f}u"
        entries.append({
            "param": pn.replace("W_", "W ").replace("L_", "L "),
            "mode": fmt_nm(mode_nm),
            "cons_frac": cons,
            "cons_txt": f"{mode_count}/{n} ({cons*100:.0f}%)",
            "range": f"[{fmt_nm(min(nm_vals))}, {fmt_nm(max(nm_vals))}]",
        })

    entries.sort(key=lambda e: e["cons_frac"], reverse=True)

    headers = ["Parameter", "Mode", "Consistency", "Range [min, max]"]
    rows = [[e["param"], e["mode"], e["cons_txt"], e["range"]] for e in entries]

    n_rows = len(rows) + 1
    fig_h = max(5, n_rows * 0.32 + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")
    fig.suptitle(f"Per-Transistor Size Consistency Across Top {len(designs)} Designs\n"
                 "Sorted by consistency: top rows = sizes the RL reliably finds "
                 "(useful regardless of other transistors)",
                 fontsize=11, fontweight="bold", y=0.99)

    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                   bbox=[0, 0, 1, 0.96], colWidths=[0.22, 0.18, 0.30, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        else:
            e = entries[r - 1]
            shade = plt.cm.RdYlGn(e["cons_frac"])
            if c == 2:
                cell.set_facecolor(shade)
                cell.set_text_props(color="#222222", fontweight="bold", fontsize=9)
            elif c == 0:
                cell.set_facecolor("#ecf0f1")
                cell.set_text_props(color="#2c3e50", fontweight="bold", ha="left", fontsize=9)
            else:
                cell.set_facecolor("#f8f9fa" if r % 2 == 0 else "white")
                cell.set_text_props(color="#333333", fontsize=9)
    if save:
        fig.savefig(f"{prefix}_fig_parameter_mode_table.png", bbox_inches="tight", dpi=140)
        print("  Saved: fig_parameter_mode_table")
    return fig

def fig_corner_bars(pvt_designs, save, prefix):
    """Grouped TT/SS/FF bars for CR and ALE, top 5 designs plus the baseline."""
    if not pvt_designs:
        print("  SKIP fig8: no 3-corner designs"); return None

    show = pvt_designs[:5]

    entries = list(show)
    labels = [f"#{i+1}" for i in range(len(show))]
    x = np.arange(len(entries))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle("PVT Corner Performance of Top AutoChaos Designs\n"
                 f"({TOPOLOGY} Circuit)", fontsize=12)

    b1 = ax1.bar(x-w, [e["tt_cr"] for e in entries], w,
                 label="TT (1.1 V / 27°C)", color=C_BLUE, edgecolor="white", linewidth=0.5)
    b2 = ax1.bar(x, [e["ss_cr"] for e in entries], w,
                 label="SS (1.045 V / 70°C)", color=C_GREEN, edgecolor="white", linewidth=0.5)
    b3 = ax1.bar(x+w, [e["ff_cr"] for e in entries], w,
                 label="FF (1.155 V / 0°C)", color=C_ORANGE, edgecolor="white", linewidth=0.5)
    ax1.axhline(TAU_CR, color="black", linewidth=1.0, linestyle="-.",
                label=f"τ_CR = {TAU_CR}")
    ax1.set_ylabel("Chaotic Ratio (CR)", fontsize=10)
    ax1.set_title("Chaotic Ratio per PVT Corner")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(0, max(e["ff_cr"] for e in entries) * 1.2)
    for bars in [b1, b2, b3]:
        for rect in bars:
            h = rect.get_height()
            if h > 0.01:
                ax1.text(rect.get_x()+rect.get_width()/2, h+0.005,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    b4 = ax2.bar(x-w, [e["tt_ale"] for e in entries], w,
                 label="TT", color=C_BLUE, edgecolor="white", linewidth=0.5)
    b5 = ax2.bar(x, [e["ss_ale"] for e in entries], w,
                 label="SS", color=C_GREEN, edgecolor="white", linewidth=0.5)
    b6 = ax2.bar(x+w, [e["ff_ale"] for e in entries], w,
                 label="FF", color=C_ORANGE, edgecolor="white", linewidth=0.5)
    ax2.axhline(TAU_ALE, color="black", linewidth=1.0, linestyle="-.",
                label=f"τ_ALE = {TAU_ALE}")
    ax2.set_ylabel("Average Lyapunov Exponent (ALE)", fontsize=10)
    ax2.set_title("ALE per PVT Corner")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    for bars in [b4, b5, b6]:
        for rect in bars:
            h = rect.get_height()
            if h > 0.005:
                ax2.text(rect.get_x()+rect.get_width()/2, h+0.002,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig8_corner_bars.png")
        print("  Saved: fig8_corner_bars")
    return fig

def fig_summary_panel(df, all_sims, by_widths, pvt_designs, save, prefix):
    """One-table run summary with the headline numbers."""
    n_iters = len(df)
    wall_h = df["time_total_s"].iloc[-1]/3600 if n_iters else 0
    n_tt = sum(1 for s in all_sims if s["process"] == "tt")
    n_ss = sum(1 for s in all_sims if s["process"] == "ss")
    n_3c = sum(1 for w in by_widths.values()
                  if "tt" in w and "ss" in w and "ff" in w)
    best_min_cr = pvt_designs[0]["min_cr"] if pvt_designs else 0
    best_nom_cr = max((s["cr"] for s in all_sims if s["process"]=="tt"), default=0)
    best_min_ale = pvt_designs[0]["min_ale"] if pvt_designs else 0
    r_mean = df["env_runners/episode_return_mean"].values
    final_reward = r_mean[-1] if len(r_mean) else 0

    metrics = [
        ("Training Iterations", f"{n_iters}"),
        ("Total Wall Time (h)", f"{wall_h:.1f}"),
        ("TT-Corner Evaluations", f"{n_tt:,}"),
        ("Full PVT Evaluations", f"{n_ss:,}"),
        ("3-Corner Confirmed Designs", f"{n_3c}"),
        ("Best Nominal CR", f"{best_nom_cr:.4f}"),
        ("Best Min CR (all PVT)", f"{best_min_cr:.4f}"),
        ("Best Min ALE (all PVT)", f"{best_min_ale:.4f}"),
    ]
    metrics += [
        ("Final Mean Reward", f"{final_reward:.4f}"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    fig.suptitle("AutoChaos Training Summary\n"
                 f"({TOPOLOGY} Chaotic PUF Circuit)",
                 fontsize=12)

    col_labels = ["Metric", "Value"]
    rows = [[m[0], m[1]] for m in metrics]
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center",
                   colWidths=[0.65, 0.25])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        else:
            cell.set_facecolor("white")
        if c == 0:
            cell.set_text_props(ha="left")

        if r > 0 and "Improvement" in rows[r-1][0]:
            cell.set_facecolor("#d4edda")
            cell.set_text_props(color="#155724", fontweight="bold")
        if r > 0 and "Best Min CR" in rows[r-1][0]:
            cell.set_facecolor("#cce5ff")
            cell.set_text_props(color="#004085", fontweight="bold")

    fig.tight_layout(rect=[0,0,1,0.92])
    if save:
        fig.savefig(f"{prefix}_fig9_summary_panel.png")
        print("  Saved: fig9_summary_panel")
    return fig

def fig_iteration_curves(df, topology_name, tau_cr, save, prefix):
    n = len(df)
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    r_mean = df["env_runners/episode_return_mean"].values if "env_runners/episode_return_mean" in df.columns else np.full(n, np.nan)
    r_max = df["env_runners/episode_return_max"].values if "env_runners/episode_return_max" in df.columns else np.full(n, np.nan)
    r_min = df["env_runners/episode_return_min"].values if "env_runners/episode_return_min" in df.columns else np.full(n, np.nan)
    has_chaos = "env_runners/chaos/cr_nominal" in df.columns
    cr_nom = df["env_runners/chaos/cr_nominal"].values if has_chaos else None
    cr_worst = df["env_runners/chaos/cr_worst"].values if has_chaos else None
    cr_nom_iter = df["env_runners/chaos/cr_nominal_iter"].values if "env_runners/chaos/cr_nominal_iter" in df.columns else None
    cr_worst_iter = df["env_runners/chaos/cr_worst_iter"].values if "env_runners/chaos/cr_worst_iter" in df.columns else None
    iw = max(3, n // 8)
    valid = r_mean[~np.isnan(r_mean)]
    trend = np.polyfit(np.arange(len(valid)), valid, 1)[0] if len(valid) >= 4 else 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Training Progress per PPO Iteration\n"
                 f"({topology_name})", fontsize=12)

    ax1.fill_between(iters, r_min, r_max, alpha=0.12, color=C_BLUE, label="Min/max range")
    ax1.plot(iters, r_mean, color=C_GREY, linewidth=0.8, alpha=0.7, label="Mean reward")
    ax1.plot(iters, smooth(r_mean, iw), color=C_BLUE, linewidth=2.2,
             label=f"Smoothed mean (trend: {trend:+.4f}/iter)")
    ax1.plot(iters, r_max, color=C_ORANGE, linewidth=1.0, linestyle="--",
             alpha=0.7, label="Best episode this iteration")
    if len(valid) >= 4:
        z = np.polyfit(np.arange(len(valid)), valid, 1)
        ax1.plot(iters, np.poly1d(z)(np.arange(len(iters))),
                 color=C_RED, linewidth=1.2, linestyle=":",
                 label=f"Linear trend")
    ax1.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    ax1.set_ylabel("Episode Reward", fontsize=10)
    ax1.set_title("Reward per Iteration")
    ax1.legend(fontsize=8, loc="upper left",
               bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax1.annotate(f"Final: {r_mean[-1]:.3f}",
                 xy=(iters[-1], r_mean[-1]),
                 xytext=(iters[max(0,len(iters)-20)], r_mean[-1] - 0.08),
                 fontsize=8, color=C_BLUE,
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.8))

    if has_chaos:
        ax2.plot(iters, cr_nom, color=C_BLUE, linewidth=1.0, alpha=0.5,
                 label="CR nominal (rolling max, w=100)")
        ax2.plot(iters, cr_worst, color=C_RED, linewidth=1.0, alpha=0.5,
                 label="CR worst corner (rolling max, w=100)")
        ax2.plot(iters, smooth(cr_nom, iw), color=C_BLUE, linewidth=2.2)
        ax2.plot(iters, smooth(cr_worst, iw), color=C_ORANGE, linewidth=2.0,
                 linestyle="--", label="CR worst (smoothed)")
        if cr_nom_iter is not None:
            ax2.plot(iters, cr_nom_iter, color=C_TEAL, linewidth=1.0,
                     linestyle=":", alpha=0.8, label="CR nominal (per-iter max, w=20)")
        if cr_worst_iter is not None:
            ax2.plot(iters, cr_worst_iter, color=C_PURPLE, linewidth=1.0,
                     linestyle=":", alpha=0.8, label="CR worst (per-iter max, w=20)")
        ax2.axhline(tau_cr, color=C_GREEN, linewidth=1.2, linestyle="-.",
                    label=f"Target τ_CR = {tau_cr}")
    else:
        ax2.text(0.5, 0.5, "AutoChaosCallbacks metrics not available in this run",
                 ha="center", va="center", transform=ax2.transAxes, color=C_GREY)
    ax2.set_xlabel("Training Iteration", fontsize=10)
    ax2.set_ylabel("Chaotic Ratio (CR)", fontsize=10)
    ax2.set_title("Chaotic Ratio per Iteration")
    ax2.legend(fontsize=8, loc="upper left",
               bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    if save:
        fig.savefig(f"{prefix}_fig10_iteration_curves.png")
        print("  Saved: fig10_iteration_curves")
    return fig

def fig_ppo_health(df, topology_name, save, prefix):
    n = len(df)
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    vf_var = df["learners/default_policy/vf_explained_var"].values if "learners/default_policy/vf_explained_var" in df.columns else None
    entropy = df["learners/default_policy/entropy"].values if "learners/default_policy/entropy" in df.columns else None
    loss = df["learners/default_policy/total_loss"].values if "learners/default_policy/total_loss" in df.columns else None
    healthy = df["fault_tolerance/num_healthy_workers"].values if "fault_tolerance/num_healthy_workers" in df.columns else None
    iter_s = df["time_this_iter_s"].values if "time_this_iter_s" in df.columns else None
    kl = df["learners/default_policy/mean_kl_loss"].values if "learners/default_policy/mean_kl_loss" in df.columns else None

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f"PPO Training Health Diagnostics\n"
                 f"({topology_name})", fontsize=12)

    def sax(ax, title, xlabel="Iteration", ylabel=""):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=8)

    if vf_var is not None:
        ax = axes[0, 0]
        ax.plot(iters, vf_var, color=C_BLUE, linewidth=1.5)
        ax.axhline(0, color=C_ORANGE, linewidth=1.0, linestyle="--", label="Zero")
        ax.axhline(0.5, color=C_GREEN, linewidth=1.0, linestyle="-.", label="Good (0.5)")
        ax.fill_between(iters, vf_var, 0, where=(np.array(vf_var)>0), alpha=0.12, color=C_GREEN)
        ax.fill_between(iters, vf_var, 0, where=(np.array(vf_var)<0), alpha=0.12, color=C_RED)
        ax.legend(fontsize=8)
        sax(ax, "Value Function Explained Variance\n(>0.5 = learning well,  <0 = not learning)",
            ylabel="VF Exp. Var.")

    if entropy is not None:
        ax = axes[0, 1]
        ax.plot(iters, entropy, color=C_PURPLE, linewidth=1.5)
        sax(ax, "Policy Entropy\n(Decreasing = converging)", ylabel="Entropy")

    if loss is not None:
        ax = axes[0, 2]
        ax.plot(iters, loss, color=C_RED, linewidth=1.5)
        sax(ax, "Total PPO Loss", ylabel="Loss")

    if healthy is not None:
        ax = axes[1, 0]
        ax.fill_between(iters, healthy, alpha=0.25, color=C_GREEN)
        ax.plot(iters, healthy, color=C_GREEN, linewidth=1.5)
        ax.set_ylim(0, None)
        sax(ax, "Active Workers\n(Drops = restarts / phantom risk)", ylabel="Count")

    if iter_s is not None:
        ax = axes[1, 1]
        ax.plot(iters, iter_s/60, color=C_GREY, linewidth=0.8, alpha=0.6, label="Raw")
        ax.plot(iters, smooth(iter_s/60, 5), color=C_ORANGE, linewidth=2.0, label="Smoothed")
        ax.legend(fontsize=8)
        sax(ax, "Iteration Time", ylabel="Minutes")

    if kl is not None:
        ax = axes[1, 2]
        ax.plot(iters, kl, color=C_TEAL, linewidth=1.5)
        sax(ax, "Mean KL Loss", ylabel="KL")
    else:
        axes[1, 2].axis("off")

    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig11_ppo_health.png")
        print("  Saved: fig11_ppo_health")
    return fig

def fig_top20_summary_table(pvt_designs, topology_name, tau_cr, tau_ale, save, prefix, top_n=20):
    designs = pvt_designs[:top_n]
    if not designs:
        print("  SKIP fig12: no 3-corner designs"); return None

    fig_h = max(6, len(designs) * 0.42 + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.suptitle(f"Top {len(designs)} PVT-Robust Designs - Summary\n"
                 f"({topology_name}, ranked by PVT reward value)",
                 fontsize=12, y=0.98)
    ax.axis("off")
    ax.set_position([0.01, 0.05, 0.98, 0.88])

    col_labels = ["Rank", "Design", "Reward", "Min CR", "Mean CR", "Min ALE", "Mean ALE", "Corners"]
    rows = []
    for i, d in enumerate(designs):
        corner_vals = d["corners"]
        mean_cr = sum(corner_vals[proc]["cr"] for proc in corner_vals) / len(corner_vals)
        rows.append([
            f"#{i+1}",
            f"D{i+1}",
            f"{d.get('reward', float('nan')):.4f}",
            f"{d['min_cr']:.4f}",
            f"{mean_cr:.4f}",
            f"{d['min_ale']:.4f}",
            f"{d['mean_ale']:.4f}",
            str(d["n_corners"]),
        ])

    n_rows = len(designs) + 1

    cell_font = max(9, min(13, int(380 / n_rows)))
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center",
                   bbox=[0, 0, 1, 1],
                   colWidths=[0.07, 0.11, 0.13, 0.13, 0.13, 0.13, 0.13, 0.11])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(cell_font)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row_idx == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
            continue
        d = designs[row_idx - 1]
        passes = d["min_cr"] >= tau_cr and d["min_ale"] >= tau_ale
        near = d["min_cr"] >= tau_cr * 0.8
        if passes:
            cell.set_facecolor("#d4edda")
            cell.set_text_props(color="#155724")
        elif near:
            cell.set_facecolor("#cce5ff")
            cell.set_text_props(color="#004085")
        else:
            cell.set_facecolor("#f8f9fa" if row_idx % 2 == 0 else "white")
            cell.set_text_props(color="#333333")
        if col_idx == 7:
            cell.set_text_props(ha="left")

    fig.text(0.01, 0.035,
             f"■ Green = Min CR ≥ {tau_cr} AND Min ALE ≥ {tau_ale}   "
             f"■ Blue = Min CR ≥ {tau_cr*0.8:.2f} (near target)   "
             f"■ White = below target",
             color="#555555", fontsize=8)

    fig.text(0.01, 0.005,
             r"Ranking reward:  $R = w_{ALE}\,s(ALE_{wc},\tau_{ALE}) + w_{CR}\,s(CR_{wc},\tau_{CR}) + B_{allpass}$   "
             r"where $X_{wc}=\alpha\min_c X_c+(1-\alpha)\,\overline{X}_c$",
             color="#333333", fontsize=8)
    if save:
        fig.savefig(f"{prefix}_fig12_top20_summary.png")
        print("  Saved: fig12_top20_summary")
    return fig

def fig_top20_per_corner(pvt_designs, topology_name, tau_cr, tau_ale, save, prefix, top_n=20):
    designs = pvt_designs[:top_n]
    if not designs:
        print("  SKIP fig13: no 3-corner designs"); return None

    corner_order = ["tt", "ss", "ff"]
    corner_labels = {
        "tt": "TT  1.1V/27°C",
        "ss": "SS  1.045V/70°C",
        "ff": "FF  1.155V/0°C",
    }

    headers = ["Rank", "Design", "Reward",
               "Min CR", "Mean CR", "Min ALE", "Mean ALE"]
    for proc in corner_order:
        headers.append(f"{corner_labels[proc]}\nCR")
        headers.append(f"{corner_labels[proc]}\nALE")

    rows = []
    cr_grid = []
    ale_grid = []

    for i, d in enumerate(designs):
        corner_vals = d["corners"]
        mean_cr = sum(corner_vals[proc]["cr"] for proc in corner_vals) / len(corner_vals)
        row = [
            f"#{i+1}",
            f"D{i+1}",
            f"{d.get('reward', float('nan')):.4f}",
            f"{d['min_cr']:.4f}",
            f"{mean_cr:.4f}",
            f"{d['min_ale']:.4f}",
            f"{d['mean_ale']:.4f}",
        ]
        rc, ra = [], []
        for proc in corner_order:
            cd = d.get(proc)
            if cd:
                row.append(f"{cd['cr']:.4f}")
                row.append(f"{cd['ale']:.4f}")
                rc.append(cd["cr"])
                ra.append(cd["ale"])
            else:
                row.append("—"); row.append("—")
                rc.append(float("nan")); ra.append(float("nan"))
        rows.append(row)
        cr_grid.append(rc)
        ale_grid.append(ra)

    n_cols = len(headers)
    n_corners = len(corner_order)

    row_h = 0.42
    head_h = 1.2
    fig_h = head_h + row_h * len(designs) + 0.6
    fig_w = 18

    rank_w = 0.05
    design_w = 0.08
    reward_w = 0.075
    summ_w = 0.075
    corn_w = 0.085
    cw = [rank_w, design_w, reward_w] + [summ_w]*4 + [corn_w]*(n_corners*2)
    total = sum(cw); cw = [w/total for w in cw]

    cell_font = 10
    params_font = 10

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Top {len(designs)} PVT-Robust Designs: Per-Corner CR and ALE\n"
        f"({topology_name}, ranked by PVT reward value, corners TT / SS / FF)",
        fontsize=12, fontweight="bold", y=1.03)
    ax.set_facecolor("white"); ax.axis("off")

    ax.set_position([0.01, 0.04, 0.98, 0.90])
    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellLoc="center",
                   bbox=[0, 0, 1, 1],
                   colWidths=cw)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(cell_font)

    corner_head_bg = {"tt": "#1a3a5c", "ss": "#1a3a1a", "ff": "#3a1a1a"}
    corner_head_fg = {"tt": "#ffffff", "ss": "#ffffff", "ff": "#ffffff"}

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            if c < 2:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontsize=cell_font, fontweight="bold")
            elif c < 6:
                cell.set_facecolor("#34495e")
                cell.set_text_props(color="white", fontsize=cell_font, fontweight="bold")
            else:
                ci = (c - 6) // 2
                proc = corner_order[ci % n_corners]
                cell.set_facecolor(corner_head_bg[proc])
                cell.set_text_props(color=corner_head_fg[proc],
                                    fontsize=cell_font-0.5, fontweight="bold")
            continue

        di = r - 1
        if c == 0:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(color="#2c3e50", fontweight="bold", fontsize=cell_font)
        elif c == 1:
            cell.set_facecolor("white")
            cell.set_text_props(color="#333333", ha="left", fontsize=params_font)
        elif c < 6:
            try:
                val = float(rows[di][c])
                if c in (2, 3):
                    if val >= tau_cr:
                        cell.set_facecolor("#d4edda"); cell.set_text_props(color="#155724", fontweight="bold", fontsize=cell_font)
                    elif val >= tau_cr * 0.8:
                        cell.set_facecolor("#cce5ff"); cell.set_text_props(color="#004085", fontsize=cell_font)
                    else:
                        cell.set_facecolor("#fff3cd"); cell.set_text_props(color="#856404", fontsize=cell_font)
                else:
                    if val >= tau_ale:
                        cell.set_facecolor("#d4edda"); cell.set_text_props(color="#155724", fontsize=cell_font)
                    else:
                        cell.set_facecolor("#fff3cd"); cell.set_text_props(color="#856404", fontsize=cell_font)
            except (ValueError, IndexError):
                cell.set_facecolor("white"); cell.set_text_props(color="#333333", fontsize=cell_font)
        else:
            co = c - 6
            ci = co // 2
            is_cr = (co % 2 == 0)
            try:
                val = cr_grid[di][ci] if is_cr else ale_grid[di][ci]
                if val != val:
                    cell.set_facecolor("#f0f0f0")
                    cell.set_text_props(color="#999999", fontsize=cell_font)
                elif is_cr:
                    if val >= tau_cr:
                        cell.set_facecolor("#d4edda"); cell.set_text_props(color="#155724", fontweight="bold", fontsize=cell_font)
                    elif val >= tau_cr * 0.7:
                        cell.set_facecolor("#fff3cd"); cell.set_text_props(color="#856404", fontsize=cell_font)
                    else:
                        cell.set_facecolor("#f8d7da"); cell.set_text_props(color="#721c24", fontsize=cell_font)
                else:
                    if val >= tau_ale:
                        cell.set_facecolor("#cce5ff"); cell.set_text_props(color="#004085", fontsize=cell_font)
                    elif val >= tau_ale * 0.75:
                        cell.set_facecolor("#fff3cd"); cell.set_text_props(color="#856404", fontsize=cell_font)
                    else:
                        cell.set_facecolor("#f8d7da"); cell.set_text_props(color="#721c24", fontsize=cell_font)
            except (IndexError, TypeError):
                cell.set_facecolor("white"); cell.set_text_props(color="#333333", fontsize=cell_font)

    fig.text(0.01, 0.005,
             "■ Green = meets target   ■ Blue = ALE meets target   "
             "■ Yellow = near target   ■ Red = below target   ■ Grey = not evaluated",
             color="#555555", fontsize=8)

    if save:
        fig.savefig(f"{prefix}_fig13_top20_per_corner.png")
        print("  Saved: fig13_top20_per_corner")
    return fig

def fig_reward_per_iteration(df, topology_name, save, prefix):
    """Reward per PPO iteration with min/max band, best-so-far, trend, and the
    first iteration at which the smoothed mean stops improving."""
    n = len(df)
    if n == 0:
        print("  SKIP fig14: empty progress.csv"); return None
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    r_mean = df["env_runners/episode_return_mean"].values if "env_runners/episode_return_mean" in df.columns else np.full(n, np.nan)
    r_max = df["env_runners/episode_return_max"].values if "env_runners/episode_return_max" in df.columns else np.full(n, np.nan)
    r_min = df["env_runners/episode_return_min"].values if "env_runners/episode_return_min" in df.columns else np.full(n, np.nan)
    r_best = running_best(np.where(np.isnan(r_mean), -np.inf, r_mean))
    iw = max(3, n // 12)
    r_sm = smooth(r_mean, iw)
    valid = r_mean[~np.isnan(r_mean)]
    trend = np.polyfit(np.arange(len(valid)), valid, 1)[0] if len(valid) >= 4 else 0.0

    sat_iter = None
    if n >= 10 and np.isfinite(r_sm[-1]):
        final = r_sm[-1]
        tol = max(0.01 * abs(final), 0.02)
        for i in range(n):
            if np.all(np.abs(np.asarray(r_sm[i:]) - final) <= tol):
                sat_iter = iters[i]; break

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.fill_between(iters, r_min, r_max, alpha=0.12, color=C_BLUE,
                    label="Per-iteration min/max")
    ax.plot(iters, r_mean, color=C_GREY, linewidth=0.8, alpha=0.6,
            label="Mean reward (per iteration)")
    ax.plot(iters, r_sm, color=C_BLUE, linewidth=2.2,
            label=f"Smoothed mean (w={iw})")
    ax.plot(iters, r_best, color=C_ORANGE, linewidth=1.6, linestyle="--",
            label="Best-so-far (per-iteration mean)")
    if len(valid) >= 4:
        z = np.polyfit(np.arange(len(valid)), valid, 1)
        ax.plot(iters, np.poly1d(z)(np.arange(len(iters))),
                color=C_RED, linewidth=1.2, linestyle=":",
                label=f"Linear trend: {trend:+.4f}/iter")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    if sat_iter is not None:
        ax.axvline(sat_iter, color=C_GREEN, linewidth=1.3, linestyle="-.",
                   label=f"Saturation - iter {int(sat_iter)}")
    ax.set_xlabel("PPO Training Iteration", fontsize=10)
    ax.set_ylabel("Episode Reward", fontsize=10)
    ax.set_title(f"AutoChaos Reward Trend per Training Iteration\n"
                 f"({topology_name} Chaotic Circuit)")
    ax.legend(loc="lower right", fontsize=8)
    if np.isfinite(r_mean[-1]):
        ax.annotate(f"Final mean: {r_mean[-1]:.3f}",
                    xy=(iters[-1], r_mean[-1]),
                    xytext=(iters[max(0, n-25)], r_mean[-1] - 0.20),
                    fontsize=8, color=C_BLUE,
                    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.8))
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig14_reward_per_iteration.png")
        print("  Saved: fig14_reward_per_iteration")
    return fig

def fig_loss_decomposition(df, topology_name, save, prefix):
    """PPO loss components with the adaptive KL and scheduled entropy coefficients."""
    n = len(df)
    if n == 0:
        print("  SKIP fig15: empty progress.csv"); return None
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    def col(name):
        return df[name].values if name in df.columns else None
    pol = col("learners/default_policy/policy_loss")
    vf = col("learners/default_policy/vf_loss")
    kl = col("learners/default_policy/mean_kl_loss")
    klc = col("learners/default_policy/curr_kl_coeff")
    entc = col("learners/default_policy/curr_entropy_coeff")
    iw = max(3, n // 12)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle(f"PPO Objective Decomposition\n({topology_name})",
                 fontsize=12)

    ax = axes[0, 0]
    if pol is not None:
        ax.plot(iters, pol, color=C_GREY, linewidth=0.8, alpha=0.6)
        ax.plot(iters, smooth(pol, iw), color=C_BLUE, linewidth=2.0)
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.set_title("Policy (Surrogate) Loss", fontsize=10)
    ax.set_xlabel("Iteration", fontsize=9); ax.set_ylabel("Policy loss", fontsize=9)

    ax = axes[0, 1]
    if vf is not None:
        ax.plot(iters, vf, color=C_GREY, linewidth=0.8, alpha=0.6)
        ax.plot(iters, smooth(vf, iw), color=C_RED, linewidth=2.0)
        ax.set_yscale("log")
    ax.set_title("Value Function Loss (log scale)", fontsize=10)
    ax.set_xlabel("Iteration", fontsize=9); ax.set_ylabel("VF loss", fontsize=9)

    ax = axes[1, 0]
    if kl is not None:
        ax.plot(iters, kl, color=C_PURPLE, linewidth=1.5, label="Mean KL")
        ax.axhline(KL_TARGET, color=C_GREEN, linewidth=1.0, linestyle="-.",
                   label=f"KL target = {KL_TARGET}")
    ax.set_title("Policy KL Divergence", fontsize=10)
    ax.set_xlabel("Iteration", fontsize=9); ax.set_ylabel("KL", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if klc is not None:
        ax.plot(iters, klc, color=C_ORANGE, linewidth=1.8, label="KL coeff (adaptive)")
    if entc is not None:
        ax2 = ax.twinx()
        ax2.plot(iters, entc, color=C_TEAL, linewidth=1.8, linestyle="--",
                 label="Entropy coeff (scheduled)")
        ax2.set_ylabel("Entropy coeff", fontsize=9, color=C_TEAL)
        ax2.tick_params(axis="y", labelcolor=C_TEAL, labelsize=8)
    ax.set_title("Adaptive / Scheduled Coefficients", fontsize=10)
    ax.set_xlabel("Iteration", fontsize=9); ax.set_ylabel("KL coeff", fontsize=9, color=C_ORANGE)
    ax.tick_params(axis="y", labelcolor=C_ORANGE, labelsize=8)
    ax.legend(fontsize=8, loc="center right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save:
        fig.savefig(f"{prefix}_fig15_loss_decomposition.png")
        print("  Saved: fig15_loss_decomposition")
    return fig

def fig_ppo_loss(df, topology_name, save, prefix):
    """Total PPO loss, identical to the fig11 panel, as a standalone figure."""
    n = len(df)
    if n == 0:
        print("  SKIP fig20: empty progress.csv"); return None
    if "learners/default_policy/total_loss" not in df.columns:
        print("  SKIP fig20: no total_loss column"); return None
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    loss = df["learners/default_policy/total_loss"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iters, loss, color=C_RED, linewidth=1.5)
    ax.set_title(f"Total PPO Loss\n({topology_name})", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig20_ppo_loss.png")
        print("  Saved: fig20_ppo_loss")
    return fig

def fig_mean_kl(df, topology_name, save, prefix):
    """Mean KL loss, identical to the fig11 panel, as a standalone figure."""
    n = len(df)
    if n == 0:
        print("  SKIP fig21: empty progress.csv"); return None
    if "learners/default_policy/mean_kl_loss" not in df.columns:
        print("  SKIP fig21: no mean_kl_loss column"); return None
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    kl = df["learners/default_policy/mean_kl_loss"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iters, kl, color=C_TEAL, linewidth=1.5)
    ax.set_title(f"Mean KL Loss\n({topology_name})", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("KL", fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(f"{prefix}_fig21_mean_kl.png")
        print("  Saved: fig21_mean_kl")
    return fig

def fig_success_and_eplen(df, topology_name, save, prefix):
    """All-corner pass rate over training, paired with mean episode length."""
    n = len(df)
    if n == 0:
        print("  SKIP fig16: empty progress.csv"); return None
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    ap = df["env_runners/chaos/all_pass_rate"].values if "env_runners/chaos/all_pass_rate" in df.columns else None
    eplen = df["env_runners/episode_len_mean"].values if "env_runners/episode_len_mean" in df.columns else None
    if ap is None and eplen is None:
        print("  SKIP fig16: no success/eplen columns"); return None
    iw = max(3, n // 12)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.suptitle(f"Task Success and Episode Length over Training\n"
                 f"({topology_name})", fontsize=12)
    if ap is not None:
        ax.plot(iters, ap, color=C_GREY, linewidth=0.8, alpha=0.5)
        ax.plot(iters, smooth(ap, iw), color=C_GREEN, linewidth=2.4,
                label="All-corner pass rate")
        ax.set_ylabel("All-PVT-corner pass rate", fontsize=10, color=C_GREEN)
        ax.tick_params(axis="y", labelcolor=C_GREEN)
        ax.set_ylim(0, 1.0)
    ax.set_xlabel("Training Iteration", fontsize=10)
    if eplen is not None:
        ax2 = ax.twinx()
        ax2.plot(iters, eplen, color=C_BLUE, linewidth=1.6, linestyle="--",
                 label="Mean episode length")
        ax2.set_ylabel("Mean episode length (steps)", fontsize=10, color=C_BLUE)
        ax2.tick_params(axis="y", labelcolor=C_BLUE)
    lines, labels = ax.get_legend_handles_labels()
    if eplen is not None:
        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2; labels += lab2
    ax.legend(lines, labels, loc="center right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(f"{prefix}_fig16_success_eplen.png")
        print("  Saved: fig16_success_eplen")
    return fig

def fig_sample_efficiency(df, all_sims, save, prefix, topology_name):
    """Best worst-corner CR as a function of cumulative environment steps."""
    n = len(df)
    if n == 0:
        print("  SKIP fig17: empty progress.csv"); return None
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    crw = df["env_runners/chaos/cr_worst"].values if "env_runners/chaos/cr_worst" in df.columns else None
    steps = df["num_env_steps_sampled_lifetime"].values if "num_env_steps_sampled_lifetime" in df.columns else iters*100
    if crw is None:
        print("  SKIP fig17: no cr_worst column"); return None
    best_crw = running_best(np.where(np.isnan(crw), -np.inf, crw))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.suptitle(f"Sample Efficiency: Best Robust CR vs Environment Steps\n"
                 f"({topology_name})", fontsize=12)
    ax.plot(steps, best_crw, color=C_BLUE, linewidth=2.2,
            label="Best worst-corner CR so far")
    ax.axhline(TAU_CR, color=C_GREEN, linewidth=1.2, linestyle="-.",
               label=f"Target tau_CR = {TAU_CR}")
    cross = np.argmax(best_crw >= TAU_CR) if np.any(best_crw >= TAU_CR) else None
    if cross:
        ax.axvline(steps[cross], color=C_ORANGE, linewidth=1.2, linestyle="--",
                   label=f"tau_CR reached @ {int(steps[cross]):,} steps")
    ax.set_xlabel("Environment Steps (cumulative)", fontsize=10)
    ax.set_ylabel("Worst-corner CR", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(f"{prefix}_fig17_sample_efficiency.png")
        print("  Saved: fig17_sample_efficiency")
    return fig

def fig_gradient_stability(df, topology_name, save, prefix):
    """Gradient global norm against the clip threshold, with explained variance."""
    n = len(df)
    if n == 0:
        print("  SKIP fig18: empty progress.csv"); return None
    iters = df["training_iteration"].values if "training_iteration" in df.columns else np.arange(1, n+1)
    gn = df["learners/default_policy/gradients_default_optimizer_global_norm"].values\
         if "learners/default_policy/gradients_default_optimizer_global_norm" in df.columns else None
    vf = df["learners/default_policy/vf_explained_var"].values\
         if "learners/default_policy/vf_explained_var" in df.columns else None
    if gn is None:
        print("  SKIP fig18: no gradient-norm column"); return None
    iw = max(3, n // 12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"Optimization Stability\n({topology_name})",
                 fontsize=12)
    ax1.plot(iters, gn, color=C_GREY, linewidth=0.8, alpha=0.6)
    ax1.plot(iters, smooth(gn, iw), color=C_BROWN, linewidth=2.0, label="Grad global norm")
    ax1.axhline(GRAD_CLIP, color=C_RED, linewidth=1.2, linestyle="--",
                label=f"grad_clip = {GRAD_CLIP}")
    ax1.set_title("Gradient Global Norm", fontsize=10)
    ax1.set_xlabel("Iteration", fontsize=9); ax1.set_ylabel("L2 norm", fontsize=9)
    ax1.legend(fontsize=8)
    if vf is not None:
        ax2.plot(iters, vf, color=C_BLUE, linewidth=1.5)
        ax2.axhline(0, color=C_ORANGE, linewidth=1.0, linestyle="--")
        ax2.axhline(0.5, color=C_GREEN, linewidth=1.0, linestyle="-.", label="Good (0.5)")
        ax2.fill_between(iters, vf, 0, where=(np.array(vf)>0), alpha=0.12, color=C_GREEN)
        ax2.fill_between(iters, vf, 0, where=(np.array(vf)<0), alpha=0.12, color=C_RED)
        ax2.set_title("Value Function Explained Variance", fontsize=10)
        ax2.set_xlabel("Iteration", fontsize=9); ax2.set_ylabel("VF exp. var.", fontsize=9)
        ax2.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save:
        fig.savefig(f"{prefix}_fig18_gradient_stability.png")
        print("  Saved: fig18_gradient_stability")
    return fig

def fig_run_summary_table(df, all_sims, by_widths, pvt_designs, lhs_pool,
                          topology_name, save, prefix):
    """One-figure run record: hyperparameters, environment settings, headline results."""
    n = len(df)
    def last(name, default=np.nan):
        return df[name].iloc[-1] if name in df.columns and n else default
    wall_h = last("time_total_s", 0) / 3600.0
    steps = last("num_env_steps_sampled_lifetime", 0)
    eps = last("env_runners/num_episodes_lifetime", 0)
    n_tt = sum(1 for s in all_sims if s["process"] == "tt")
    best_wc = pvt_designs[0]["min_cr"] if pvt_designs else float("nan")
    best_wa = pvt_designs[0]["min_ale"] if pvt_designs else float("nan")
    n_allpass = sum(1 for d in pvt_designs
                    if d["min_cr"] >= TAU_CR and d["min_ale"] >= TAU_ALE)

    rows = [
        ("EXPERIMENT", ""),
        ("Topology", topology_name),
        ("PPO HYPERPARAMETERS", ""),
        ("Train batch / minibatch / epochs", "100 / 40 / 10"),
        ("Learning rate", "3e-4"),
        ("gamma / lambda (GAE)", "0.99 / 0.95"),
        ("Clip param", "0.2"),
        ("KL coeff init / target", f"0.2 / {KL_TARGET}"),
        ("VF clip / VF loss coeff", "10.0 / 0.5"),
        ("Grad clip", f"{GRAD_CLIP}"),
        ("Entropy coeff schedule", "0.05 -> 0.005 over 15k steps"),
        ("Network (fcnet)", "[256, 256] relu, vf not shared"),
        ("Seed", "42"),
        ("ENVIRONMENT / REWARD", ""),
        ("Max episode steps", "5"),
        ("tau_CR / tau_ALE", f"{TAU_CR} / {TAU_ALE}"),
        ("w_ALE / w_CR / alpha", "0.5 / 0.5 / 0.6"),
        ("Success bonus / penalty", "2.0 / 1.0"),
        ("Progressive gate", f"{PROG_GATE}"),
        ("LHS pool (M / N kept)", f"500 / {len(lhs_pool)}"),
        ("RESULTS", ""),
        ("Iterations (wall-clock)", f"{n}  ({wall_h:.1f} h)"),
        ("Environment steps", f"{int(steps):,}"),
        ("Episodes", f"{int(eps):,}"),
        ("Unique TT simulations", f"{n_tt:,}"),
        ("3-corner verified designs", f"{len(pvt_designs)}"),
        ("Distinct all-pass designs", f"{n_allpass}"),
        ("Best worst-corner CR", f"{best_wc:.4f}"),
        ("Best worst-corner ALE", f"{best_wa:.4f}"),
        ("Final mean return", f"{last('env_runners/episode_return_mean'):.3f}"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    fig.suptitle(f"Run Configuration & Summary\n({topology_name})",
                 fontsize=13, y=0.98)
    y = 0.96; dy = 0.96 / (len(rows) + 1)
    for k, v in rows:
        is_header = (v == "")
        if is_header:
            ax.text(0.04, y, k, fontsize=10.5, fontweight="bold", color=C_BLUE,
                    transform=ax.transAxes, va="top")
        else:
            ax.text(0.07, y, k, fontsize=9.5, color="#222222",
                    transform=ax.transAxes, va="top")
            ax.text(0.62, y, str(v), fontsize=9.5, color="#000000",
                    transform=ax.transAxes, va="top", fontweight="medium")
        y -= dy
    if save:
        fig.savefig(f"{prefix}_table2_run_summary.png")
        print("  Saved: table2_run_summary")
    return fig

def _parse_spectre_val(tok):
    tok = tok.strip().strip("()")
    m = re.match(r"([\d.]+)([a-z]?)", tok, re.IGNORECASE)
    if not m:
        return None
    mult = {"": 1, "n": 1e-9, "u": 1e-6, "p": 1e-12, "m": 1e-3, "f": 1e-15}
    return float(m.group(1)) * mult.get(m.group(2).lower(), 1)

def _is_device_line(line):
    if re.search(r"g45[np]1lvt", line):
        return True
    if re.match(r"\s*M\S*\s", line) and re.search(r"\b[np]mos\b", line, re.IGNORECASE):
        return True
    return False

def compute_design_area(params, netlist_text):
    tunable = 0.0
    total = 0.0
    for line in netlist_text.splitlines():
        if not _is_device_line(line):
            continue
        wm = re.search(r"\bw\s*=\s*(\S+)", line, re.IGNORECASE)
        lm = re.search(r"\bl\s*=\s*(\S+)", line, re.IGNORECASE)
        mm = re.search(r"\bm\s*=\s*\(?(\d+)", line, re.IGNORECASE)
        if not (wm and lm):
            continue
        mult = int(mm.group(1)) if mm else 1
        wref = re.search(r"(W_\w+)", wm.group(1))
        lref = re.search(r"(L_\w+)", lm.group(1))
        w = float(params[wref.group(1)]) if (wref and wref.group(1) in params) else _parse_spectre_val(wm.group(1))
        l = float(params[lref.group(1)]) if (lref and lref.group(1) in params) else _parse_spectre_val(lm.group(1))
        if w is None or l is None:
            continue
        area = w * l * mult
        total += area
        if wref:
            tunable += area
    return tunable, total

def fig_design_sizing(pvt_designs, topology_name, save, prefix, top_n=10, netlist_text=None):
    """Full W/L sizing for the top-N designs, one column per design."""
    designs = pvt_designs[:top_n]
    if not designs:
        print("  SKIP fig19: no designs"); return None

    def _parse(wk):
        return dict(p.split("=") for p in wk.split("|")
                    if "=" in p and not any(c in p for c in ("PROCESS","VDD","TEMP")))

    pnames = set()
    for d in designs:
        pnames.update(_parse(d["widths"]).keys())
    w_names = sorted([n for n in pnames if n.startswith("W_")])
    l_names = sorted([n for n in pnames if n.startswith("L_")])
    param_order = w_names + l_names
    if not param_order:
        print("  SKIP fig19: no parameters"); return None

    def fmt_val(v):
        val = float(v) * 1e9
        return f"{val:.0f}n" if val < 1000 else f"{val/1000:.2f}u"

    headers = ["Parameter"] + [f"D{i+1}" for i in range(len(designs))]
    rows = []
    for pn in param_order:
        row = [pn.replace("W_", "W ").replace("L_", "L ")]
        for d in designs:
            parts = _parse(d["widths"])
            row.append(fmt_val(parts[pn]) if pn in parts else "—")
        rows.append(row)
    rows.append(["— worst ALE —"] + [f"{d['min_ale']:.3f}" for d in designs])
    if netlist_text:
        tun_row = ["— tunable area (um2) —"]
        tot_row = ["— total area (um2) —"]
        for d in designs:
            dparams = {}
            for pair in d["widths"].split("|"):
                if "=" in pair:
                    pk, pv = pair.split("=")
                    if pk not in ("PROCESS", "VDD", "TEMP"):
                        try:
                            dparams[pk] = float(pv)
                        except ValueError:
                            pass
            tun, tot = compute_design_area(dparams, netlist_text)
            tun_row.append(f"{tun * 1e12:.3f}")
            tot_row.append(f"{tot * 1e12:.3f}")
        rows.append(tun_row)
        rows.append(tot_row)
    else:
        print("  [fig19] netlist_text not provided -> area rows skipped")

    n_rows = len(rows) + 1
    n_cols = len(headers)
    fig_h = max(6, n_rows * 0.33 + 1.5)
    fig_w = max(8, 1.6 + n_cols * 1.05)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.suptitle(f"Top {len(designs)} Design Sizings (full W/L)\n"
                 f"({topology_name}) - companion to ranking tables",
                 fontsize=12, fontweight="bold", y=0.99)

    cw = [0.16] + [(0.84/len(designs))] * len(designs)
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                   bbox=[0, 0, 1, 0.97], colWidths=cw)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        elif c == 0:

            is_len = rows[r-1][0].startswith("L ")
            is_summ = rows[r-1][0].startswith("—")
            cell.set_facecolor("#34495e" if is_summ else ("#e8eef2" if is_len else "#ecf0f1"))
            cell.set_text_props(color="white" if is_summ else "#2c3e50",
                                fontweight="bold", ha="left", fontsize=9)
        else:
            is_summ = rows[r-1][0].startswith("—")
            cell.set_facecolor("#eaf6ea" if is_summ else ("#f8f9fa" if r % 2 == 0 else "white"))
            cell.set_text_props(color="#333333", fontsize=8.5,
                                fontweight="bold" if is_summ else "normal")
    if save:
        fig.savefig(f"{prefix}_fig19_design_sizing.png", bbox_inches="tight")
        print("  Saved: fig19_design_sizing")
    return fig

def main():
    global TAU_CR, TAU_ALE, W_ALE_R, W_CR_R, ALPHA_R, ROBUST_CREDIT_R
    parser = argparse.ArgumentParser(
        description="AutoChaos run analysis and figures"
    )
    parser.add_argument("--run_dir", type=str, default=".",
                        help="Directory containing progress.csv and result.json")
    parser.add_argument("--cache", type=str, default=None,
                        help="Cache file (.db or .json). Auto-detects if omitted.")
    parser.add_argument("--lhs", type=str, default="lhs_pool.json")
    parser.add_argument("--save", action="store_true",
                        help="Save figures as PNG (300 dpi)")
    parser.add_argument("--prefix", type=str, default="autochaos_3t",
                        help="Filename prefix for saved figures")
    parser.add_argument("--top", type=int, default=20,
                        help="Top N designs for parameter distribution plot")
    parser.add_argument("--top20", type=int, default=20,
                        help="Top N designs for summary and per-corner tables")
    parser.add_argument("--netlist", type=str, default=None,
                        help="Path to the netlist file for area-row computation in fig19")
    parser.add_argument("--topology", type=str, default=None,
                        help="Topology name shown in figure titles (e.g. '3-Transistor')")
    args = parser.parse_args()

    if args.cache is None:
        args.cache = _find_cache_file(args.run_dir)
        if args.cache is None:
            args.cache = _find_cache_file(".")
        if args.cache is None:
            print("WARNING: No cache file found (metrics_cache.db or metrics_cache.json)")
            args.cache = "metrics_cache.db"

    print(f"[AutoChaos Analyzer]")
    print(f"  Run dir: {args.run_dir}")
    print(f"  Cache: {args.cache} ({'SQLite' if args.cache.endswith('.db') else 'JSON'})")
    print(f"  LHS pool: {args.lhs}")
    print(f"  Save: {args.save}")

    df = load_progress(args.run_dir)
    results = load_results(args.run_dir)
    cache = load_cache(args.cache)
    lhs_pool = load_lhs(args.lhs)

    _map_cfg = None
    try:
        import yaml as _yaml0, os as _os0
        _mp0 = (results[0].get("config", {}).get("env_config", {}).get("map_config_path", "")
                if results else "")
        for _c0 in [_mp0, _os0.path.join(args.run_dir, _mp0) if _mp0 else "",
                    _os0.path.join(args.run_dir, _os0.path.basename(_mp0)) if _mp0 else "",
                    _os0.path.join(args.run_dir, "map_config_mscmi.yaml"),
                    _os0.path.join(args.run_dir, "map_config_3t.yaml")]:
            if _c0 and _os0.path.isfile(_c0):
                _map_cfg = _yaml0.safe_load(open(_c0)); break
        if _map_cfg and isinstance(_map_cfg.get("pvt_reward"), dict):
            _pr = _map_cfg["pvt_reward"]
            TAU_CR = float(_pr.get("tau_CR", TAU_CR))
            TAU_ALE = float(_pr.get("tau_ALE", TAU_ALE))

            W_ALE_R = float(_pr.get("w_ALE", W_ALE_R))
            W_CR_R = float(_pr.get("w_CR", W_CR_R))
            ALPHA_R = float(_pr.get("alpha", ALPHA_R))
            ROBUST_CREDIT_R = float(_pr.get("robust_credit", ROBUST_CREDIT_R))
            print(f"  [thresholds] from map_config: tau_CR={TAU_CR}, tau_ALE={TAU_ALE}")
            print(f"  [reward params] w_ALE={W_ALE_R}, w_CR={W_CR_R}, alpha={ALPHA_R}, robust_credit={ROBUST_CREDIT_R}")
    except Exception as _e0:
        print(f"  [thresholds] map_config not read ({_e0}); using defaults "
              f"tau_CR={TAU_CR}, tau_ALE={TAU_ALE}")

    all_sims, by_widths = parse_cache(cache)
    pvt_designs = get_pvt_designs(by_widths)
    export_top_designs_json(pvt_designs, args.prefix, top_n=10)

    _netlist_text = None
    try:
        import os as _osN
        _tpl = _map_cfg.get("netlist_template", "") if _map_cfg else ""
        _cands = []
        if args.netlist:
            _cands.append(args.netlist)
            _cands.append(_osN.path.join(args.run_dir, args.netlist))
        if _tpl:
            _cands.append(_tpl)
            _cands.append(_osN.path.join(args.run_dir, _tpl))
            _cands.append(_osN.path.join(args.run_dir, _osN.path.basename(_tpl)))
            _cands.append(_osN.path.join(args.run_dir, "templates", _osN.path.basename(_tpl)))
        for _cand in _cands:
            if _cand and _osN.path.isfile(_cand):
                _netlist_text = open(_cand).read()
                print(f"  [area] netlist loaded from {_cand}")
                break
        if not _netlist_text:
            print("  [area] netlist not found; area rows skipped in fig19")
            print("  [area] pass --netlist <path> or add netlist_template to the map config")
            if _cands:
                print(f"  [area] checked: {[c for c in _cands if c]}")
    except Exception as _eN:
        print(f"  [area] could not load netlist ({_eN}); area rows skipped")

    topology_name = args.topology or "3-Transistor"
    env_cfg = results[0].get("config", {}).get("env_config", {}) if results else {}
    mp = str(env_cfg.get("map_config_path", "")).lower()

    if "mscmi" in mp and args.topology is None:
        topology_name = "MSCMI"

    global TOPOLOGY
    TOPOLOGY = topology_name

    n_iters = len(df)
    wall_h = df["time_total_s"].iloc[-1]/3600 if n_iters else 0
    n_tt = sum(1 for s in all_sims if s["process"] == "tt")

    print(f"\n  Topology: {topology_name}")
    print(f"  Iterations: {n_iters} ({wall_h:.1f}h)")
    print(f"  TT simulations: {n_tt:,}")
    print(f"  3-corner designs: {len(pvt_designs)}")
    if pvt_designs:
        print(f"  Best min CR: {pvt_designs[0]['min_cr']:.4f}")
        print(f"  Best min ALE: {pvt_designs[0]['min_ale']:.4f}")
    print(f"  LHS entries: {len(lhs_pool)}")
    print(f"\n  Generating figures...")

    figs = []
    figs.append(("Reward Curve",
                 fig_reward_curve(df, args.save, args.prefix, topology_name)))
    figs.append(("Search Saturation",
                 fig_search_saturation(all_sims, args.save, args.prefix, topology_name)))
    figs.append(("PVT Funnel",
                 fig_pvt_funnel(all_sims, by_widths, args.save, args.prefix)))
    figs.append(("Nominal vs Worst",
                 fig_nominal_vs_worst(pvt_designs, args.save, args.prefix)))
    figs.append(("CR vs ALE Tradeoff",
                 fig_cr_ale_tradeoff(pvt_designs, args.save, args.prefix)))
    figs.append(("Distribution Shift",
                 fig_distribution_shift(all_sims, lhs_pool, args.save, args.prefix)))
    figs.append(("Parameter Dist",
                 fig_parameter_distribution(all_sims, by_widths, args.save, args.prefix, top_n=args.top)))
    figs.append(("Parameter Mode",
                 fig_parameter_mode(pvt_designs, args.save, args.prefix, top_n=args.top20)))
    figs.append(("Parameter Mode Table",
                 fig_parameter_mode_table(pvt_designs, args.save, args.prefix, top_n=args.top20)))
    figs.append(("Corner Bars",
                 fig_corner_bars(pvt_designs, args.save, args.prefix)))
    figs.append(("Summary Panel",
                 fig_summary_panel(df, all_sims, by_widths, pvt_designs, args.save, args.prefix)))
    figs.append(("Iteration Curves",
                 fig_iteration_curves(df, topology_name, TAU_CR, args.save, args.prefix)))
    figs.append(("PPO Health",
                 fig_ppo_health(df, topology_name, args.save, args.prefix)))
    figs.append(("Top 20 Summary Table",
                 fig_top20_summary_table(pvt_designs, topology_name,
                                         TAU_CR, TAU_ALE, args.save, args.prefix,
                                         top_n=args.top20)))
    figs.append(("Top 20 Per-Corner Table",
                 fig_top20_per_corner(pvt_designs, topology_name,
                                      TAU_CR, TAU_ALE, args.save, args.prefix,
                                      top_n=args.top20)))
    figs.append(("Reward per Iteration",
                 fig_reward_per_iteration(df, topology_name, args.save, args.prefix)))
    figs.append(("Loss Decomposition",
                 fig_loss_decomposition(df, topology_name, args.save, args.prefix)))
    figs.append(("PPO Policy Loss (standalone)",
                 fig_ppo_loss(df, topology_name, args.save, args.prefix)))
    figs.append(("Mean KL (standalone)",
                 fig_mean_kl(df, topology_name, args.save, args.prefix)))
    figs.append(("Success Rate & Episode Length",
                 fig_success_and_eplen(df, topology_name, args.save, args.prefix)))
    figs.append(("Sample Efficiency",
                 fig_sample_efficiency(df, all_sims, args.save, args.prefix, topology_name)))
    figs.append(("Gradient Stability",
                 fig_gradient_stability(df, topology_name, args.save, args.prefix)))
    figs.append(("Run Summary Table",
                 fig_run_summary_table(df, all_sims, by_widths, pvt_designs, lhs_pool,
                                       topology_name, args.save, args.prefix)))
    figs.append(("Design Sizing Appendix",
                 fig_design_sizing(pvt_designs, topology_name, args.save, args.prefix,
                                   top_n=min(10, args.top20 if hasattr(args,'top20') else 10),
                                   netlist_text=_netlist_text)))

    valid_figs = [f for _, f in figs if f is not None]
    if args.save:
        plt.close("all")
        print(f"\n  Done. {len(valid_figs)} figures saved with prefix '{args.prefix}'")
        print(f"  Figures:")
        for name, fig in figs:
            if fig is not None:
                print(f"    {name}")
    else:
        print(f"\n  Opening {len(valid_figs)} figures...")
        plt.show()

if __name__ == "__main__":
    main()