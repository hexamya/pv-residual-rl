"""Acceptance-aware baselines (R1) and location invariance of the peak term (R2).

R1  Acceptance-aware baselines. Is the PPO advantage explained by spending the
    budget where acceptance is high, rather than by network-aware placement?
      accept-greedy : offer the budget to neighbourhoods in decreasing order of
                      mean acceptance, each up to its expected headroom
                      (headroom / mean acceptance); the whole budget is spent.
      max-exp-cap   : myopic water-filling that maximises the capacity expected
                      to be installed this year, E[sum_i min(o_i xi_i, H_i)],
                      with xi_i ~ Beta(a_i, b_i); the whole budget is spent.
    Neither rule uses the network model, the CF, demand forecasts or training.
    Evaluated on the same paired scenarios as the paper (nominal 10000-10099,
    OAT 40000-40049, Bass 10000-10099) and compared with the stored results.

R2  Return decomposition into its peak, transformer and voltage terms, and a
    check that the peak term depends only on total installed capacity.

Usage: python scripts/16_review_checks.py
Outputs: results/review_*.csv, results/review_r2_location.json
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pvplan import pvgis
from pvplan.baselines import (AcceptGreedyPolicy, MaxExpectedCapacityPolicy,
                              PeakFirstPolicy)
from pvplan.environment import N_BUS, EnvConfig, PlanningCore
from pvplan.evaluation import evaluate_policy
from pvplan.stats import bootstrap_ci
from pvplan.uncertainty import UncertaintyConfig

RESULTS = ROOT / "results"
SEEDS_NOM = [10_000 + k for k in range(100)]
SEEDS_OAT = [40_000 + k for k in range(50)]
W_TR, W_V = 0.5, 0.5


NEW_POLICIES = {"accept-greedy": AcceptGreedyPolicy,
                "max-exp-cap": MaxExpectedCapacityPolicy}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    lo, hi = bootstrap_ci(d)
    p = 1.0 if np.allclose(d, 0) else float(sps.wilcoxon(a, b).pvalue)
    return {"mean": a.mean(), "mean_ref": b.mean(), "mean_diff": d.mean(),
            "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "n": len(d)}


def seed_average(df: pd.DataFrame, label_col: str = "algo") -> pd.DataFrame:
    """Average RL seeds per scenario (as in 06_evaluate_all.py)."""
    base = df[df[label_col].isna()] if label_col in df else df
    frames = [base]
    if label_col in df:
        for algo, g in df[df[label_col].notna()].groupby(label_col):
            agg = g.groupby("scenario_seed").mean(numeric_only=True).reset_index()
            agg["policy"] = algo
            frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def compare(new: pd.DataFrame, stored: pd.DataFrame, refs: list[str],
            label: str) -> pd.DataFrame:
    rows = []
    for pol in NEW_POLICIES:
        x = new[new.policy == pol].set_index("scenario_seed")["return_sum"]
        for ref in refs:
            y = stored[stored.policy == ref].set_index("scenario_seed")["return_sum"]
            idx = x.index.intersection(y.index)
            rows.append({"setting": label, "policy": pol, "reference": ref,
                         **paired(x.loc[idx].values, y.loc[idx].values)})
    return pd.DataFrame(rows)


def run_r1(pv_days: np.ndarray) -> None:
    sens = importlib.import_module("07_sensitivity")

    # --- reproducibility check: stored results must be regenerated exactly ---
    nominal = pd.read_csv(RESULTS / "evaluation_nominal.csv")
    pf = evaluate_policy("peak-first", PeakFirstPolicy(), pv_days, EnvConfig(),
                         SEEDS_NOM)
    stored_pf = nominal[nominal.policy == "peak-first"].set_index("scenario_seed")
    err = np.abs(pf.set_index("scenario_seed")["return_sum"]
                 - stored_pf["return_sum"]).max()
    print(f"[repro] peak-first max |return diff| vs stored = {err:.2e}", flush=True)
    assert err < 1e-9, "environment no longer reproduces the stored results"

    # --- nominal ----------------------------------------------------------
    frames = [evaluate_policy(n, cls(), pv_days, EnvConfig(), SEEDS_NOM)
              for n, cls in NEW_POLICIES.items()]
    new_nom = pd.concat(frames, ignore_index=True)
    new_nom.to_csv(RESULTS / "review_r1_nominal.csv", index=False)
    stored = seed_average(nominal)
    stats = [compare(new_nom, stored,
                     ["ppo", "sac", "rolling-lp", "peak-first", "oracle"], "nominal")]

    summary = pd.concat([stored, new_nom], ignore_index=True).groupby("policy").agg(
        return_mean=("return_sum", "mean"),
        dpeak_mean_pct=("dpeak_frac_mean", lambda s: 100 * s.mean()),
        dpeak_final_pct=("dpeak_frac_final", lambda s: 100 * s.mean()),
        installed_mw=("installed_total_mw", "mean"),
        offered_mw=("offered_total_mw", "mean"),
    ).sort_values("return_mean", ascending=False)
    summary.to_csv(RESULTS / "review_r1_summary.csv")
    print("\n=== R1 nominal summary ===\n", summary.round(4).to_string(), flush=True)

    # --- OAT robustness settings (stored PPO frozen / retrained / LP) --------
    oat_stored = pd.read_csv(RESULTS / "sensitivity_paired.csv")
    oat_stored["value"] = oat_stored["value"].fillna(-1)
    oat_frames = []
    configs = [("nominal", -1)] + [(ax, v) for ax, vals in sens.AXES.items()
                                    for v in vals if v != sens.NOMINAL.get(ax)]
    for axis, v in configs:
        cfg = (sens.build_cfg("bess_share", 0.0) if axis == "nominal"
               else sens.build_cfg(axis, v))
        new = pd.concat([evaluate_policy(n, cls(), pv_days, cfg, SEEDS_OAT)
                         for n, cls in NEW_POLICIES.items()], ignore_index=True)
        new["axis"], new["value"] = axis, v
        oat_frames.append(new)
        st = oat_stored[(oat_stored.axis == axis)
                        & np.isclose(oat_stored.value.astype(float), float(v))]
        refs = [r for r in ["ppo-frozen", "ppo-retrained", "rolling-lp"]
                if r in set(st.policy)]
        stats.append(compare(new, st, refs, f"{axis}={v}"))
        print(f"[oat] {axis}={v} done", flush=True)
    pd.concat(oat_frames, ignore_index=True).to_csv(
        RESULTS / "review_r1_oat.csv", index=False)

    # --- Bass-type (peer-effect) acceptance -----------------------------------
    cfg_bass = EnvConfig(uncertainty=UncertaintyConfig(adoption_model="bass"))
    new_bass = pd.concat([evaluate_policy(n, cls(), pv_days, cfg_bass, SEEDS_NOM)
                          for n, cls in NEW_POLICIES.items()], ignore_index=True)
    new_bass.to_csv(RESULTS / "review_r1_bass.csv", index=False)
    bass = seed_average(pd.read_csv(RESULTS / "bass_ablation.csv"))
    stats.append(compare(new_bass, bass,
                         ["ppo-bass", "ppo-nominal-in-bass", "rolling-lp"], "bass"))

    out = pd.concat(stats, ignore_index=True)
    out.to_csv(RESULTS / "review_r1_stats.csv", index=False)
    print("\n=== R1 paired comparisons (new policy minus reference) ===")
    print(out.round(4).to_string(index=False), flush=True)


# ---------------------------------------------------------------------------
# R2: decomposition and location invariance
# ---------------------------------------------------------------------------
def decompose(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tr_term"] = -W_TR * out["overload_delta_sum"]
    out["v_term"] = -W_V * out["volt_delta_sum"]
    out["peak_term"] = out["return_sum"] - out["tr_term"] - out["v_term"]
    return out


def run_r2(pv_days: np.ndarray) -> None:
    rows = []
    sources = {
        "nominal": seed_average(pd.read_csv(RESULTS / "evaluation_nominal.csv")),
        "bass": seed_average(pd.read_csv(RESULTS / "bass_ablation.csv")),
    }
    for f in ["review_r1_nominal.csv", "review_r1_bass.csv"]:
        if (RESULTS / f).exists():
            key = "nominal" if "nominal" in f else "bass"
            sources[key] = pd.concat([sources[key], pd.read_csv(RESULTS / f)])
    sp = pd.read_csv(RESULTS / "sensitivity_paired.csv")
    for (axis, v), g in sp.groupby(["axis", sp["value"].fillna(-1)]):
        sources[f"oat:{axis}={v}"] = g
    if (RESULTS / "review_r1_oat.csv").exists():
        ro = pd.read_csv(RESULTS / "review_r1_oat.csv")
        for (axis, v), g in ro.groupby(["axis", "value"]):
            key = f"oat:{axis}={v}"
            sources[key] = pd.concat([sources.get(key, g.iloc[0:0]), g])

    for name, df in sources.items():
        d = decompose(df)
        for pol, g in d.groupby("policy"):
            rows.append({"setting": name, "policy": pol,
                         "return": g["return_sum"].mean(),
                         "peak_term": g["peak_term"].mean(),
                         "tr_term": g["tr_term"].mean(),
                         "v_term": g["v_term"].mean(),
                         "installed_mw": g["installed_total_mw"].mean()})
    dec = pd.DataFrame(rows)
    dec.to_csv(RESULTS / "review_r2_decomposition.csv", index=False)
    show = dec[dec.setting.isin(["nominal", "bass", "oat:impedance_scale=1.0"])]
    print("\n=== R2 return decomposition ===")
    print(show.round(4).to_string(index=False), flush=True)

    # (a) direct test: same scenario draws, same total MW, different locations
    core = PlanningCore(pv_days, EnvConfig(), seed=0)
    rng = np.random.default_rng(1)
    res = {}
    for total in (0.5, 1.0, 1.4):
        peaks, volts, trs = [], [], []
        for _ in range(200):
            x, rem = np.zeros(N_BUS), total            # same total, feasible mix
            for i in rng.permutation(N_BUS):
                x[i] = min(core.pot_mw[i], rem * rng.uniform())
                rem -= x[i]
            for i in rng.permutation(N_BUS):
                add = min(core.pot_mw[i] - x[i], rem)
                x[i], rem = x[i] + add, rem - add
            core.reseed(10_000)
            core.reset_state()
            core.installed_mw = x
            info = core.apply_year(np.zeros(N_BUS), 0.0)
            peaks.append(info["dpeak_frac"])
            volts.append(info["volt_delta"])
            trs.append(info["overload_delta"])
        res[f"total_{total}MW"] = {
            "dpeak_frac_range": float(np.ptp(peaks)),
            "volt_delta_range": float(np.ptp(volts)),
            "overload_delta_range": float(np.ptp(trs)),
            "dpeak_frac_mean": float(np.mean(peaks)),
        }

    # (b) stored runs: within each scenario, final-year dpeak vs installed MW
    nom = pd.read_csv(RESULTS / "evaluation_nominal.csv")
    rhos, viol, n_pairs = [], 0, 0
    for _, g in nom.groupby("scenario_seed"):
        x, y = g["installed_total_mw"].values, g["dpeak_frac_final"].values
        rhos.append(sps.spearmanr(x, y).statistic)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        mask = dx > 1e-9
        n_pairs += int(mask.sum())
        viol += int((dy[mask] < -1e-12).sum())    # more MW but smaller reduction
    res["stored_runs"] = {
        "n_runs_per_scenario": int(nom.groupby("scenario_seed").size().iloc[0]),
        "spearman_median": float(np.median(rhos)),
        "spearman_min": float(np.min(rhos)),
        "ordered_pairs": n_pairs,
        "pairs_more_MW_but_less_reduction": viol,
    }
    with open(RESULTS / "review_r2_location.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== R2 location invariance ===")
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    pv = pvgis.load_summer_days()
    if "--r2-only" not in sys.argv:
        run_r1(pv)
    run_r2(pv)
