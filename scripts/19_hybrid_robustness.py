"""Robustness of the hybrid/residual PPO on the weak feeder (no retraining).

One-at-a-time perturbations of the weak-feeder case, as in 07_sensitivity.py
(BESS share, demand growth, horizon, budget, rooftop potential), plus the
Bass-type acceptance model. Policies trained on the nominal weak feeder are
evaluated frozen; rule-based policies and LPs are re-run in each setting.
100 paired test scenarios (10000-10099) per setting.

Usage: python scripts/19_hybrid_robustness.py [--workers 7]
Outputs: results/robust_evaluation.csv, robust_summary.csv, robust_stats.csv
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

from pvplan import pvgis
from pvplan.environment import EnvConfig
from pvplan.uncertainty import UncertaintyConfig

RESULTS = ROOT / "results"
SEEDS = [10_000 + k for k in range(100)]
SEED_IDS = range(5)


def settings() -> dict[str, EnvConfig]:
    base = EnvConfig(impedance_scale=1.0)
    out = {"nominal": base}
    for v in (0.1, 0.2, 0.3):
        out[f"bess_share={v}"] = dataclasses.replace(base, bess_share=v)
    for v in (0.01, 0.02, 0.03, 0.04):
        out[f"growth_mean={v}"] = dataclasses.replace(
            base, uncertainty=UncertaintyConfig(growth_mean=v))
    for v in (5, 15):
        out[f"horizon_years={v}"] = dataclasses.replace(base, horizon_years=v)
    for v in (0.125, 0.375):
        out[f"annual_budget_mw={v}"] = dataclasses.replace(base, annual_budget_mw=v)
    for v in (1.5, 2.0):
        out[f"potential_scale={v}"] = dataclasses.replace(base, potential_scale=v)
    out["bass"] = dataclasses.replace(
        base, uncertainty=UncertaintyConfig(adoption_model="bass"))
    return out


def _job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    weak = importlib.import_module("17_weak_feeder")
    hyb = importlib.import_module("18_hybrid")
    setting, label = job
    tuned = json.loads((RESULTS / "weak_tuned.json").read_text())
    family, _, seed = label.rpartition("-s")
    if family in hyb.VARIANTS:
        pol = hyb.HybridPolicy(family, int(seed))
    elif family == "ppo":
        pol = weak.make_policy(label)
    elif label == "max-exp-cap-V":
        pol = weak.make_policy("max-exp-cap", tuned["kappa"])
    elif label == "max-exp-cap":
        pol = weak.make_policy("max-exp-cap", 0.0)
    elif label == "rolling-lp-V":
        pol = weak.make_policy("rolling-lp-v", tuned["lam_v"])
    else:
        pol = weak.make_policy(label)
    df = evaluate_policy(label, pol, pvgis.load_summer_days(), settings()[setting], SEEDS)
    df["setting"] = setting
    print(f"  {setting:22s} {label:14s} {df['return_sum'].mean():.4f}", flush=True)
    return df


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    labels_slow = ["rolling-lp-V"]
    labels_fast = ([f"{f}-s{s}" for f in ("residual", "hybrid", "ppo") for s in SEED_IDS]
                   + ["max-exp-cap-V", "max-exp-cap", "rolling-lp"])
    names = list(settings())
    jobs = [(n, l) for l in labels_slow for n in names] + \
           [(n, l) for n in names for l in labels_fast]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        df = pd.concat(list(ex.map(_job, jobs)), ignore_index=True)
    df.to_csv(RESULTS / "robust_evaluation.csv", index=False)

    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    comp = (df.groupby(["setting", "family", "scenario_seed"])
            .mean(numeric_only=True).reset_index())
    comp["v_term"] = -0.5 * comp["volt_delta_sum"]
    summary = comp.pivot_table(index="setting", columns="family",
                               values="return_sum", aggfunc="mean").loc[names]
    summary.to_csv(RESULTS / "robust_summary.csv")

    rows = []
    for setting in names:
        c = comp[comp.setting == setting]
        for fam in ("residual", "hybrid"):
            x = c[c.family == fam].set_index("scenario_seed")["return_sum"]
            for ref in ("max-exp-cap-V", "rolling-lp-V", "ppo"):
                y = c[c.family == ref].set_index("scenario_seed")["return_sum"].loc[x.index]
                d = x.values - y.values
                lo, hi = bootstrap_ci(d)
                rows.append({"setting": setting, "policy": fam, "reference": ref,
                             "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue),
                             "wins": int((d > 0).sum()), "n": len(d)})
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "robust_stats.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== mean return by setting ===\n",
          summary[["residual", "hybrid", "max-exp-cap-V", "ppo", "rolling-lp-V",
                   "max-exp-cap", "rolling-lp"]].round(4).to_string())
    print("\n=== residual / hybrid minus reference ===\n",
          stats.round(4).to_string(index=False))
    for fam in ("residual", "hybrid"):
        for ref in ("max-exp-cap-V", "rolling-lp-V", "ppo"):
            s = stats[(stats.policy == fam) & (stats.reference == ref)
                      & (stats.setting != "nominal")]
            print(f"{fam} vs {ref}: better {int((s.ci_lo > 0).sum())}, "
                  f"tie {int(((s.ci_lo <= 0) & (s.ci_hi >= 0)).sum())}, "
                  f"worse {int((s.ci_hi < 0).sum())} of {len(s)}")


if __name__ == "__main__":
    main()
