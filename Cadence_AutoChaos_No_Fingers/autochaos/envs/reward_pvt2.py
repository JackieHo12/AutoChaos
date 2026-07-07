
import math
from typing import Dict, List, Tuple

DEFAULT_CORNERS = [
    (1.100, 27.0, "nominal", "tt"),
    (1.045, 70.0, "slow",    "ss"),
    (1.155, 0.0,  "fast",    "ff"),
]

# Alias so any external code importing CORNERS still works.
CORNERS = DEFAULT_CORNERS
DEFAULT_TAU_ALE = 0.35
DEFAULT_TAU_CR = 0.95
DEFAULT_W_ALE = 0.5
DEFAULT_W_CR = 0.5
DEFAULT_ALPHA = 0.6
DEFAULT_BONUS = 2.0
DEFAULT_NOM_BONUS = 0.5
DEFAULT_PENALTY = 1.0
DEFAULT_ROBUST_CREDIT = 1.0


def _score(value, tau):
    v = max(float(value), 0.0)
    if tau <= 0.0: tau = 1e-9
    if v < tau:
        return 0.5 * (v / tau)
    excess = (v - tau) / tau
    return 0.5 + 0.3 * math.tanh(excess) + 0.2 * min(excess, 1.0)


def _safe(x):
    try: x = float(x)
    except Exception: return 0.0
    if math.isnan(x) or math.isinf(x): return 0.0
    return max(0.0, x)


def _corner_reward(ale, cr, tau_ale, tau_cr, w_ale, w_cr):
    s_ale = _score(ale, tau_ale)
    s_cr = _score(cr,  tau_cr)
    r_c = w_ale * s_ale + w_cr * s_cr
    return r_c, {"ALE": ale, "CR": cr, "S_ALE": s_ale, "S_CR": s_cr, "R_c": r_c}


def compute_pvt_reward(corner_metrics, tau_ale=DEFAULT_TAU_ALE, tau_cr=DEFAULT_TAU_CR,
                       w_ale=DEFAULT_W_ALE, w_cr=DEFAULT_W_CR, alpha=DEFAULT_ALPHA,
                       bonus=DEFAULT_BONUS, penalty=DEFAULT_PENALTY,
                       nom_bonus=DEFAULT_NOM_BONUS, robust_credit=DEFAULT_ROBUST_CREDIT, corner_names=None):
    n = len(corner_metrics)
    if n == 0:
        return -penalty, {"ERROR": "no corner metrics provided"}
    corner_rewards, corner_debug = [], []
    all_pass, all_dead = True, True
    ales, crs = [], []  # all corners (debug/back-compat)
    valid_ales, valid_crs = [], []  # corners with a real simulation result
    n_failed = 0
    for i, cm in enumerate(corner_metrics):
        ale = _safe(cm.get("ALE", 0.0))
        cr  = _safe(cm.get("chaotic_ratio", cm.get("CR", 0.0)))
        ales.append(ale)
        crs.append(cr)
        _sim_failed = bool(cm.get("sim_failed", False))
        r_c, dbg_c = _corner_reward(ale, cr, tau_ale, tau_cr, w_ale, w_cr)
        if _sim_failed:
            n_failed += 1
        else:
            corner_rewards.append(r_c)
            valid_ales.append(ale)
            valid_crs.append(cr)
        if corner_names and i < len(corner_names):
            cname = str(corner_names[i])
        elif i < len(CORNERS):
            cname = CORNERS[i][2]
        else:
            cname = f"corner{i}"
        if _sim_failed:
            dbg_c = dict(dbg_c); dbg_c["SIM_FAILED"] = True
        corner_debug.append({cname: dbg_c})
        if _sim_failed or ale < tau_ale or cr < tau_cr: all_pass = False
        if (not _sim_failed) and (ale > 0.0 or cr > 0.0): all_dead = False
    if not valid_ales:
        dbg = {"corner_rewards": [], "corner_details": corner_debug,
               "R_min": 0.0, "R_mean": 0.0, "R_robust": 0.0,
               "ale_wc": 0.0, "cr_wc": 0.0,
               "all_pass": False, "all_dead": False,
               "bonus_applied": 0.0, "nom_bonus_applied": 0.0,
               "penalty_applied": 0.0, "proximity_multiplier": 1.0,
               "progress": 0.0, "n_failed_corners": n_failed,
               "REWARD": -0.1 * penalty,
               "tau_ALE": tau_ale, "tau_CR": tau_cr, "alpha": alpha,
               "ALL_CORNERS_SIM_FAILED": True}
        return float(-0.1 * penalty), dbg
    n_valid = len(valid_ales)
    ale_wc = alpha * min(valid_ales) + (1.0 - alpha) * (sum(valid_ales) / n_valid)
    cr_wc = alpha * min(valid_crs)  + (1.0 - alpha) * (sum(valid_crs)  / n_valid)
    s_ale_wc = _score(ale_wc, tau_ale)
    s_cr_wc = _score(cr_wc,  tau_cr)
    r_robust = w_ale * s_ale_wc + w_cr * s_cr_wc
    r_min = min(corner_rewards)
    r_mean = sum(corner_rewards) / len(corner_rewards)
    r_bonus = bonus     if all_pass else 0.0
    r_nom_bonus = nom_bonus if (n == 1 and (crs[0] if crs else 0.0) >= 1.0 and (ales[0] if ales else 0.0) >= tau_ale) else 0.0
    nom_cr = crs[0] if crs else 0.0
    proximity_multiplier = 1.0
    if all_dead:
        reward = -penalty
        progress = 0.0
    elif not all_pass:
        progress = min(nom_cr / tau_cr, 1.0) if tau_cr > 0 else 1.0
        reward = -1.0 + progress + robust_credit * r_robust
    else:
        progress = 1.0
        reward = r_robust + r_bonus + r_nom_bonus
    dbg = {"corner_rewards": corner_rewards, "corner_details": corner_debug,
           "R_min": r_min, "R_mean": r_mean, "R_robust": r_robust,
           "ale_wc": ale_wc, "cr_wc": cr_wc,
           "all_pass": all_pass, "all_dead": all_dead,
           "n_failed_corners": n_failed,
           "bonus_applied": r_bonus, "nom_bonus_applied": r_nom_bonus,
           "penalty_applied": penalty if all_dead else 0.0,
           "proximity_multiplier": proximity_multiplier,
           "progress": progress,
           "REWARD": reward, "tau_ALE": tau_ale, "tau_CR": tau_cr, "alpha": alpha}
    return float(reward), dbg


def format_pvt_log(dbg):
    lines = []
    for cd in dbg.get("corner_details", []):
        for cname, cv in cd.items():
            lines.append(f"{cname}: ALE={cv['ALE']:.4f} CR={cv['CR']:.4f} Rc={cv['R_c']:.3f}")
    summary = (f"R_min={dbg['R_min']:.4f} R_mean={dbg['R_mean']:.4f} "
               f"nom_bonus={dbg['nom_bonus_applied']:.1f} "
               f"penalty={dbg['penalty_applied']:.1f} "
               f"progress={dbg.get('progress',0.0):.3f} "
               + (f"FAILED_CORNERS={dbg['n_failed_corners']} " if dbg.get('n_failed_corners') else "") +
               f"all_pass={dbg['all_pass']} "
               f"=> REWARD={dbg['REWARD']:.4f}")
    return " | ".join(lines) + " || " + summary
