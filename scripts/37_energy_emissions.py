"""Energy, emission and budget-efficiency consequences of the main policies.

The reward measures summer peak and voltage relief only. For an energy-
management reader the same episodes also determine how much PV energy the
programme delivers, how much grid electricity (and CO2) it displaces and how
much of the subsidised capacity is actually installed. Every test episode of
the main policies is replayed with identical draws (as in script 27) and the
feeder's installed capacity after each annual allocation is recorded.

Energy: the installed capacity of year t generates for one year with the
mean annual specific yield of the PVGIS record used by the environment
(1-kWp building-integrated c-Si, optimal tilt, 14% losses, Tehran, SARAH3
2005-2023). No reverse power flow occurs in any simulated summer hour, so the
PV output is taken as fully absorbed by the feeder load. Emissions: IFI
default combined-margin grid factor for Iran, intermittent renewables
(570 gCO2/kWh, IFI Dataset of Default Grid Factors v2.0, 2019).

Usage: python scripts/37_energy_emissions.py [--scenarios 100] [--workers 7]
Outputs: results/energy_emissions.csv (per scenario and policy),
         results/energy_stats.csv (paired differences),
         results/energy_summary.json (means and constants)
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results"
EF_T_PER_MWH = 0.570          # IFI v2.0 combined margin, Iran, intermittent RE
POLICIES = {
    "stiff": ["max-exp-cap", "accept-greedy", "rolling-lp", "peak-first"]
             + [f"ppo-s{s}" for s in range(5)],
    "weak": ["max-exp-cap-V", "max-exp-cap", "rolling-lp-V"]
            + [f"ppo-s{s}" for s in range(5)] + [f"residual-s{s}" for s in range(5)]
            + [f"hybrid-s{s}" for s in range(5)],
}
IMPEDANCE = {"stiff": 0.5, "weak": 1.0}
PAIRS = {
    "stiff": [("max-exp-cap", "ppo"), ("max-exp-cap", "rolling-lp"), ("ppo", "rolling-lp"),
              ("max-exp-cap", "accept-greedy"), ("max-exp-cap", "peak-first")],
    "weak": [("residual", "max-exp-cap-V"), ("hybrid", "max-exp-cap-V"), ("ppo", "max-exp-cap-V"),
             ("max-exp-cap-V", "max-exp-cap"), ("rolling-lp-V", "max-exp-cap-V"),
             ("residual", "ppo")],
}
METRICS = ["installed_final_mw", "energy_programme_gwh", "energy_final_year_mwh",
           "co2_programme_t", "budget_efficiency", "energy_per_mw_offered_mwh"]


def annual_yield_kwh_per_kwp() -> tuple[float, float, int]:
    """Mean and SD of the annual PV yield over the full years of the PVGIS record."""
    raw = json.loads((ROOT / "data" / "raw" / "pvgis_tehran_2005_2023.json").read_text())
    by_year: dict[str, float] = {}
    for rec in raw["outputs"]["hourly"]:
        by_year[rec["time"][:4]] = by_year.get(rec["time"][:4], 0.0) + rec["P"] / 1000.0
    v = np.array(list(by_year.values()))
    return float(v.mean()), float(v.std(ddof=1)), len(v)


def make(feeder: str, label: str):
    exact = importlib.import_module("27_opendss_exact_return")
    if label.startswith("hybrid-s"):
        hyb = importlib.import_module("18_hybrid")
        return hyb.HybridPolicy("hybrid", int(label[-1]))
    return exact.make(feeder, label)


def _scenario(job: tuple) -> list[dict]:
    import torch
    from pvplan.environment import EnvConfig, PVDeploymentEnv
    from pvplan import pvgis
    torch.set_num_threads(1)
    feeder, scenario, y_kwh = job
    cfg = EnvConfig(impedance_scale=IMPEDANCE[feeder])
    days = pvgis.load_summer_days()
    rows = []
    for label in POLICIES[feeder]:
        policy = make(feeder, label)
        env = PVDeploymentEnv(days, cfg)
        obs, _ = env.reset(options={"scenario_seed": scenario})
        inst, offered, rpf, done = [], 0.0, 0.0, False
        while not done:
            obs, _, done, _, info = env.step(policy(env.core, obs))
            inst.append(float(env.core.installed_mw.sum()))
            offered += float(info.get("offered_mw", 0.0))
            rpf = max(rpf, float(info.get("reverse_flow_mw", 0.0)))
        inst = np.array(inst)
        e_years = inst * y_kwh                           # MWh per year (MW x kWh/kWp)
        row = {"feeder": feeder, "scenario_seed": scenario, "policy": label,
               "installed_final_mw": inst[-1],
               "energy_programme_gwh": e_years.sum() / 1000.0,
               "energy_final_year_mwh": e_years[-1],
               "co2_programme_t": e_years.sum() * EF_T_PER_MWH,
               "offered_total_mw": offered, "rpf_max_mw": rpf}
        row.update({f"installed_y{t + 1}_mw": x for t, x in enumerate(inst)})
        rows.append(row)
    return rows


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    y_kwh, y_sd, n_years = annual_yield_kwh_per_kwp()
    jobs = [(f, 10_000 + k, y_kwh) for f in POLICIES for k in range(args.scenarios)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        df = pd.DataFrame([r for rows in ex.map(_scenario, jobs) for r in rows])
    df["budget_efficiency"] = df.installed_final_mw / df.offered_total_mw
    df["energy_per_mw_offered_mwh"] = 1000.0 * df.energy_programme_gwh / df.offered_total_mw
    df.to_csv(RESULTS / "energy_emissions.csv", index=False)

    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    comp = df.groupby(["feeder", "family", "scenario_seed"]).mean(numeric_only=True).reset_index()
    rows = []
    for feeder, plist in PAIRS.items():
        c = comp[comp.feeder == feeder]
        for a, b in plist:
            for col in METRICS:
                x = c[c.family == a].set_index("scenario_seed")[col].sort_index()
                y = c[c.family == b].set_index("scenario_seed")[col].loc[x.index]
                d = (x - y).values
                lo, hi = bootstrap_ci(d)
                rows.append({"feeder": feeder, "policy": a, "reference": b, "metric": col,
                             "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": 1.0 if np.allclose(d, 0)
                             else float(sps.wilcoxon(x.values, y.values).pvalue),
                             "wins": int((d > 0).sum()),
                             "rrb": rank_biserial_paired(x.values, y.values)})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "energy_stats.csv", index=False)

    ycols = [f"installed_y{t + 1}_mw" for t in range(10)]
    means = comp.groupby(["feeder", "family"])[METRICS + ["offered_total_mw", "rpf_max_mw"] + ycols].mean()
    summary = {
        "yield_kwh_per_kwp": y_kwh, "yield_sd_kwh_per_kwp": y_sd, "yield_years": n_years,
        "ef_t_per_mwh": EF_T_PER_MWH, "n_scenarios": args.scenarios,
        "rpf_max_mw_any": float(df.rpf_max_mw.max()),
        "means": {f"{f}|{p}": {k: float(v) for k, v in r.items()} for (f, p), r in means.iterrows()},
        "pairs": {f"{r.feeder}|{r.policy}|{r.reference}|{r.metric}":
                  {"diff": r["diff"], "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "wilcoxon_p": r.wilcoxon_p,
                   "wins": int(r.wins), "rrb": r.rrb} for _, r in st.iterrows()},
    }
    (RESULTS / "energy_summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 220)
    print(f"yield {y_kwh:.1f} kWh/kWp/yr (SD {y_sd:.1f}, {n_years} years); max reverse flow {df.rpf_max_mw.max():.3f} MW")
    print(means[METRICS].round(3).to_string())
    print(st[st.metric.isin(["energy_programme_gwh", "co2_programme_t"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
