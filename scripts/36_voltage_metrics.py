"""Magnitude-based voltage metric, thresholds and OLTC details.

The reward counts bus-hours below 0.95 pu. This script re-runs the test
episodes (common random numbers, so returns are unchanged) and reports, for
the same policies, the PV-attributable reduction of
  - the undervoltage deficit sum max(0, 0.95 - V) over bus-hours (pu*h per summer),
  - the count of bus-hours below 0.94 and below 0.96 pu,
on the nominal weak feeder and along the OLTC sweep of D29. It also records the
OLTC tap operations, a fixed-tap variant and the sensitivity to the LDC
setpoint, and how the undervoltage bus-hours without PV are distributed over
PV and non-PV hours.

Usage:
  python scripts/36_voltage_metrics.py run [--workers 6]
  python scripts/36_voltage_metrics.py extra [--workers 6]
  python scripts/36_voltage_metrics.py timing
  python scripts/36_voltage_metrics.py overvoltage
  python scripts/36_voltage_metrics.py stats
Outputs: results/vmetric_evaluation.csv, vmetric_stats.csv, vmetric_levels.csv, vmetric_dominance.json,
         uv_timing.json, oltc_overvoltage.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results"
SEEDS = [10_000 + k for k in range(100)]
T = 10
BH = 32 * 24 * 92                     # bus-hours per 92-day summer
SETTINGS = ["none", "1.01", "1.02", "1.03", "1.04", "1.05"]


def _reg():
    return importlib.import_module("33_voltage_regulation")


def _job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan import pvgis
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    setting, label, seeds, variant = job
    reg = _reg()
    cfg = reg.cfg(setting)
    if variant.startswith("fixed"):
        cfg = dataclasses.replace(cfg, oltc_mode="fixed")
    elif variant.startswith("vset="):
        cfg = dataclasses.replace(cfg, oltc_vset=float(variant.split("=")[1]))
    if label == "hybrid" or label.startswith("hybrid-s"):
        pol = importlib.import_module("18_hybrid").HybridPolicy("hybrid", int(label[-1]))
    else:
        pol = reg._policy(setting, label)
    df = evaluate_policy(label, pol, pvgis.load_summer_days(), cfg, seeds)
    df["setting"], df["variant"] = setting, variant
    return df


def cmd_run(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    base = (["max-exp-cap", "max-exp-cap-V retuned"]
            + [f"residual-s{k}" for k in range(5)] + [f"ppo-s{k}" for k in range(5)])
    jobs = [(s, lab, SEEDS, "ldc") for s in SETTINGS for lab in base]
    jobs += [("none", f"hybrid-s{k}", SEEDS, "ldc") for k in range(5)]
    jobs += [("1.02", f"{k}-s{i}", SEEDS, "ldc") for k in ("residual-rt", "ppo-rt") for i in range(3)]
    jobs += [(s, "rolling-lp-V", SEEDS[i:i + 25], "ldc") for s in SETTINGS for i in range(0, 100, 25)]
    # OLTC details: fixed tap and LDC setpoint, rules only
    for s in SETTINGS[1:]:
        for lab in ("max-exp-cap", "max-exp-cap-V retuned"):
            jobs.append((s, lab, SEEDS, "fixed"))
            if s in ("1.02", "1.03"):
                jobs += [(s, lab, SEEDS, f"vset={v}") for v in (0.99, 1.01)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        df = pd.concat(list(ex.map(_job, jobs)), ignore_index=True)
    df.to_csv(RESULTS / "vmetric_evaluation.csv", index=False)
    print("rows", len(df))


EXTRA = ["ppo-nominal", "accept-greedy", "peak-first", "random", "rolling-lp"]


def _extra_job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan import pvgis
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    label, seeds = job
    pol = importlib.import_module("17_weak_feeder").make_policy(label)
    df = evaluate_policy(label, pol, pvgis.load_summer_days(), _reg().cfg("none"), seeds)
    df["setting"], df["variant"] = "none", "ldc"
    return df


def cmd_extra(workers: int) -> None:
    """The remaining policies of the weak-feeder table (nominal feeder), appended."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(lab, SEEDS) for lab in EXTRA if lab != "rolling-lp"]
    jobs += [("rolling-lp", SEEDS[i:i + 25]) for i in range(0, 100, 25)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        new = pd.concat(list(ex.map(_extra_job, jobs)), ignore_index=True)
    df = pd.read_csv(RESULTS / "vmetric_evaluation.csv")
    df = pd.concat([df[~((df.setting.astype(str) == "none") & df.policy.isin(EXTRA))], new],
                   ignore_index=True)
    df.to_csv(RESULTS / "vmetric_evaluation.csv", index=False)
    print("rows", len(df))


def cmd_timing() -> None:
    """Undervoltage bus-hours without PV on the weak feeder's test scenarios:
    share in hours with PV output and share that the full rooftop potential could lift."""
    from pvplan.network import Network33, V_MIN_PU
    from pvplan.profiles import bus_shapes, rho_potential
    ex = importlib.import_module("35_lookahead_bound")
    net = Network33(impedance_scale=1.0)
    shapes = bus_shapes()
    pot = rho_potential() * net.p_load_kw / 1000.0
    lift_unit = net.r_common_pu @ pot                         # pu per unit PV output, full potential
    tot = in_pv = in_pv30 = liftable = 0
    for sc in SEEDS:
        for y in ex.record_draws("weak", sc):
            g = net.p_load_kw[:, None, None] * shapes[:, None, :] * y["load_scale"] * y["noise"]
            q = net.q_load_kvar[:, None, None] * shapes[:, None, :] * y["load_scale"] * y["noise"]
            v = net.voltages_pu(g.reshape(32, -1), q.reshape(32, -1)).reshape(32, *g.shape[1:])
            uv = v < V_MIN_PU
            days = y["days"][None, :, :]
            tot += uv.sum()
            in_pv += (uv & (days > 0)).sum()
            in_pv30 += (uv & (days > 0.3)).sum()
            liftable += (uv & (lift_unit[:, None, None] * days >= V_MIN_PU - v)).sum()
    out = {"uv_bus_hours_sampled": int(tot), "share_in_pv_hours": in_pv / tot,
           "share_in_pv_hours_over_30pct": in_pv30 / tot, "share_liftable_full_potential": liftable / tot}
    (RESULTS / "uv_timing.json").write_text(json.dumps(out, indent=2))
    print(out)


def cmd_overvoltage() -> None:
    """Worst-case voltage rise by PV for each OLTC limit: final installed state of the
    unweighted rule on 30 test scenarios, then the full rooftop potential installed,
    70% load and the clearest-hour PV of the record."""
    from pvplan import pvgis
    from pvplan.baselines import MaxExpectedCapacityPolicy
    from pvplan.environment import PVDeploymentEnv
    pv = pvgis.load_summer_days()
    out = {}
    for setting in SETTINGS:
        env = PVDeploymentEnv(pv, _reg().cfg(setting))
        pol, vmax = MaxExpectedCapacityPolicy(0.0), 0.0
        for sc in SEEDS[:30]:
            obs, _ = env.reset(options={"scenario_seed": sc})
            done = False
            while not done:
                obs, _, done, _, _ = env.step(pol(env.core, obs))
            c = env.core
            p = (c.p_nom_kw[:, None] * c.shapes * c.load_scale * 0.7
                 - c.pot_mw[:, None] * 1000 * pv.max(axis=0)[None, :])
            q = c.q_nom_kvar[:, None] * c.shapes * c.load_scale * 0.7
            vmax = max(vmax, float(c.net.voltages_pu(p, q).max()))
        out[setting] = vmax
    (RESULTS / "oltc_overvoltage.json").write_text(json.dumps(out, indent=2))
    print(out)


def cmd_stats() -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci
    df = pd.read_csv(RESULTS / "vmetric_evaluation.csv")
    df["setting"] = df.setting.astype(str)
    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    df["count095_bh"] = -df.volt_delta_sum / T * BH
    df["count094_bh"] = -df.uv094_delta_sum / T * BH
    df["count096_bh"] = -df.uv096_delta_sum / T * BH
    df["deficit_puh"] = -df.uv_deficit_delta_sum / T * BH          # pu*h relieved per summer
    df["deficit_gross_puh"] = df.uv_deficit_gross_sum / T * BH
    metrics = ["return_sum", "count095_bh", "count094_bh", "count096_bh", "deficit_puh"]

    # consistency with earlier runs (common random numbers)
    old = pd.read_csv(RESULTS / "oltc_evaluation.csv")
    old["setting"] = old.setting.astype(str)
    m = df[df.variant == "ldc"].merge(old, on=["setting", "policy", "scenario_seed"], suffixes=("", "_old"))
    print("max |return - earlier run|:", float((m.return_sum - m.return_sum_old).abs().max()))

    fam = df.groupby(["variant", "setting", "family", "scenario_seed"])[metrics + ["deficit_gross_puh", "tap_ops_per_day"]].mean()
    fam.groupby(level=[0, 1, 2]).mean().reset_index().to_csv(RESULTS / "vmetric_levels.csv", index=False)
    R = "max-exp-cap-V retuned"
    pairs = [(R, "max-exp-cap"), ("residual", R), ("hybrid", R), ("ppo", R), ("rolling-lp-V", R),
             ("residual-rt", R), ("ppo-rt", R)]
    rows = []
    for (variant, setting), g in fam.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        fams = set(g.index.get_level_values(0))
        for a, b in pairs:
            if a not in fams or b not in fams:
                continue
            x, y = g.loc[a], g.loc[b].loc[g.loc[a].index]
            row = {"variant": variant, "setting": setting, "policy": a, "reference": b,
                   "deficit_gross_puh": float(g.loc[b].deficit_gross_puh.mean()),
                   "tap_ops_per_day": float(g.loc[b].tap_ops_per_day.mean())}
            for mtr in metrics:
                d = (x[mtr] - y[mtr]).values
                lo, hi = bootstrap_ci(d)
                row.update({f"{mtr}": float(d.mean()), f"{mtr}_lo": lo, f"{mtr}_hi": hi,
                            f"{mtr}_wins": int((d > 0).sum()),
                            f"{mtr}_p": 1.0 if np.allclose(d, 0) else float(sps.wilcoxon(d).pvalue),
                            f"{mtr}_ref": float(y[mtr].mean())})
            rows.append(row)
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "vmetric_stats.csv", index=False)

    # weak dominance of the residual agent over the rule on the nominal weak feeder:
    # return difference as a function of the voltage weight w_v (peak weight 1)
    g = fam.loc[("ldc", "none")]
    base = df[(df.variant == "ldc") & (df.setting == "none")].groupby(["family", "scenario_seed"])[
        ["dpeak_frac_mean", "volt_delta_sum"]].mean()
    x, y = base.loc["residual"], base.loc[R].loc[base.loc["residual"].index]
    dp = (T * (x.dpeak_frac_mean - y.dpeak_frac_mean)).values
    dv = (-(x.volt_delta_sum - y.volt_delta_sum)).values
    dom = {"peak_term_diff": float(dp.mean()), "peak_term_ci": list(bootstrap_ci(dp)), "by_weight": {}}
    for w in (0.05, 0.1, 0.2, 0.5, 1.0):
        lo, hi = bootstrap_ci(dp + w * dv)
        dom["by_weight"][str(w)] = {"diff": float((dp + w * dv).mean()), "ci_lo": lo, "ci_hi": hi}
    dom["min_significant_weight"] = min(float(w) for w, o in dom["by_weight"].items() if o["ci_lo"] > 0)
    (RESULTS / "vmetric_dominance.json").write_text(json.dumps(dom, indent=2))
    print("dominance:", dom["min_significant_weight"], dom["peak_term_ci"])
    pd.set_option("display.width", 250)
    show = ["variant", "setting", "policy", "reference", "return_sum", "count095_bh", "count094_bh",
            "count096_bh", "deficit_puh", "deficit_puh_lo", "deficit_puh_hi", "deficit_puh_ref"]
    print(st[show].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "extra", "timing", "overvoltage", "stats"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"run": lambda: cmd_run(a.workers), "extra": lambda: cmd_extra(a.workers),
     "timing": cmd_timing, "overvoltage": cmd_overvoltage, "stats": cmd_stats}[a.cmd]()
