"""Structured policies with a misspecified acceptance model.

The rules, the allocation layer of the residual agent and the LP-V normally use
the simulator's true class-level Beta parameters. Here they plan with a
believed model, while the environment keeps drawing from the true one.

Belief cases:
  pessimistic    all class means x0.75
  optimistic     all class means x1.25
  class-errors   residential x1.25, mixed x0.85, commercial x0.80, admin x1.15
                 (the believed ranking of classes changes)
  dispersion     true means, concentration x0.5 (believes acceptance is more uncertain)
Plain PPO does not use the acceptance model; its stored results are the reference.

Usage: python scripts/28_acceptance_misspec.py [--workers 3]
Outputs: results/misspec_evaluation.csv, misspec_stats.csv
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
SEEDS = [10_000 + k for k in range(100)]
CASES = {
    "pessimistic": ({"residential": 0.75, "mixed": 0.75, "commercial": 0.75, "admin": 0.75}, 1.0),
    "optimistic": ({"residential": 1.25, "mixed": 1.25, "commercial": 1.25, "admin": 1.25}, 1.0),
    "class-errors": ({"residential": 1.25, "mixed": 0.85, "commercial": 0.80, "admin": 1.15}, 1.0),
    "dispersion": ({"residential": 1.0, "mixed": 1.0, "commercial": 1.0, "admin": 1.0}, 0.5),
}
LABELS = {"stiff": ["max-exp-cap", "accept-greedy"],
          "weak": ["max-exp-cap-V", "max-exp-cap", "rolling-lp-V"]
                  + [f"residual-s{s}" for s in range(5)]}


def install_belief(case: str) -> None:
    """Replace the acceptance model used by policies (not by the environment)."""
    import pvplan.allocation as alloc
    import pvplan.baselines as base
    import pvplan.environment as envm
    from pvplan.profiles import class_labels
    factors, conc_scale = CASES[case]
    f = np.array([factors[c] for c in class_labels()])
    true_params = alloc.acceptance_params

    def believed(core):
        a, b = true_params(core)
        conc = (a + b) * conc_scale
        mean = np.clip(a / (a + b) * f[:len(a)], 0.02, 0.98)
        return mean * conc, (1 - mean) * conc

    base.acceptance_params = believed
    envm.acceptance_params = believed


def _job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan import pvgis
    from pvplan.baselines import AcceptGreedyPolicy, MaxExpectedCapacityPolicy, RollingLPVPolicy
    from pvplan.environment import EnvConfig
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    case, feeder, label = job
    install_belief(case)
    hyb = importlib.import_module("18_hybrid")
    tuned = json.loads((RESULTS / "weak_tuned.json").read_text())
    if label.startswith("residual-s"):
        pol = hyb.HybridPolicy("residual", int(label[-1]))
    else:
        pol = {"max-exp-cap": lambda: MaxExpectedCapacityPolicy(0.0),
               "max-exp-cap-V": lambda: MaxExpectedCapacityPolicy(tuned["kappa"]),
               "accept-greedy": AcceptGreedyPolicy,
               "rolling-lp-V": lambda: RollingLPVPolicy(tuned["lam_v"])}[label]()
    cfg = EnvConfig(impedance_scale=0.5 if feeder == "stiff" else 1.0)
    df = evaluate_policy(label, pol, pvgis.load_summer_days(), cfg, SEEDS)
    df["case"], df["feeder"] = case, feeder
    print(f"  {case:12s} {feeder:5s} {label:14s} {df['return_sum'].mean():.4f}", flush=True)
    return df


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--exclude", nargs="*", default=[], help="labels to skip (e.g. rolling-lp-V)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these labels and merge")
    args = ap.parse_args()
    jobs = [(c, f, l) for c in CASES for f in LABELS for l in LABELS[f]
            if l not in args.exclude and (args.only is None or l in args.only)]
    jobs.sort(key=lambda j: j[2] != "rolling-lp-V")          # slow LP jobs first
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        df = pd.concat(list(ex.map(_job, jobs)), ignore_index=True)
    path = RESULTS / "misspec_evaluation.csv"
    if args.only is not None and path.exists():          # merge a partial rerun
        old = pd.read_csv(path)
        df = pd.concat([old[~old.policy.isin(df.policy.unique())], df], ignore_index=True)
    df.to_csv(path, index=False)

    # references with the true model and plain PPO (stored, same scenarios)
    stiff_ref = pd.concat([pd.read_csv(RESULTS / "evaluation_nominal.csv"),
                           pd.read_csv(RESULTS / "review_r1_nominal.csv")])
    weak_ref = pd.read_csv(RESULTS / "hybrid_evaluation.csv")
    fam = lambda d: d.assign(family=d.policy.str.replace(r"-s\d$", "", regex=True)).groupby(
        ["family", "scenario_seed"])["return_sum"].mean()
    ref = {"stiff": fam(stiff_ref), "weak": fam(weak_ref)}
    rows = []
    for (case, feeder), g in df.groupby(["case", "feeder"]):
        mis = fam(g)
        comps = ([("max-exp-cap", "ppo"), ("accept-greedy", "ppo")] if feeder == "stiff" else
                 [("residual", "max-exp-cap-V"), ("max-exp-cap-V", "ppo"), ("residual", "ppo"),
                  ("rolling-lp-V", "max-exp-cap-V")])
        for a, b in comps:
            if a not in mis.index.get_level_values(0):
                continue
            x = mis.loc[a]
            y = (mis.loc[b] if b in mis.index.get_level_values(0) else ref[feeder].loc[b]).loc[x.index]
            d = (x - y).values
            lo, hi = bootstrap_ci(d)
            rows.append({"case": case, "feeder": feeder, "policy": a + " (believed)",
                         "reference": b + (" (believed)" if b in mis.index.get_level_values(0) else ""),
                         "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue),
                         "wins": int((d > 0).sum()), "rrb": rank_biserial_paired(x.values, y.values)})
        for a in mis.index.get_level_values(0).unique():      # cost of misspecification
            x, y = mis.loc[a], ref[feeder].loc[a].loc[mis.loc[a].index]
            d = (x - y).values
            lo, hi = bootstrap_ci(d)
            rows.append({"case": case, "feeder": feeder, "policy": a + " (believed)",
                         "reference": a + " (true model)", "diff": d.mean(), "ci_lo": lo,
                         "ci_hi": hi, "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue)
                         if not np.allclose(d, 0) else 1.0,
                         "wins": int((d > 0).sum()), "rrb": rank_biserial_paired(x.values, y.values)})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "misspec_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
