"""A rule that learns acceptance from observed uptake (follow-up to D26).

AdaptiveCapacityPolicy starts from the (possibly wrong) believed class means
and updates them each year from the offers and installations it observes. The
prior strength n0 (MW of offers) is chosen on the tuning scenarios
(30000-30049) by minimax regret over five beliefs (true + the four cases of
28_acceptance_misspec.py), relative to the unhedged rule with the true model.
The test uses 100 scenarios on both feeders (stiff: kappa = 0; weak: kappa = 0.5).

Usage: python scripts/31_adaptive_rule.py [--workers 6]
Outputs: results/adaptive_tuning.csv, adaptive_choice.json, adaptive_evaluation.csv, adaptive_stats.csv
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
SEEDS_TUNE = [30_000 + k for k in range(50)]
SEEDS_TEST = [10_000 + k for k in range(100)]
N0S = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
BELIEFS = ["true", "pessimistic", "optimistic", "class-errors", "dispersion"]
KAPPA = {"stiff": 0.0, "weak": 0.5}


def _job(job: tuple) -> pd.DataFrame:
    from pvplan import pvgis
    from pvplan.baselines import AdaptiveCapacityPolicy
    from pvplan.environment import EnvConfig
    from pvplan.evaluation import evaluate_policy
    feeder, belief, n0, seeds = job
    if belief != "true":
        importlib.import_module("28_acceptance_misspec").install_belief(belief)
    cfg = EnvConfig(impedance_scale=0.5 if feeder == "stiff" else 1.0)
    df = evaluate_policy(f"adaptive n0={n0}", AdaptiveCapacityPolicy(KAPPA[feeder], n0),
                         pvgis.load_summer_days(), cfg, seeds)
    df["feeder"], df["belief"], df["n0"] = feeder, belief, n0
    return df


def run(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(_job, jobs)), ignore_index=True)


def main() -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    hedge_tune = pd.read_csv(RESULTS / "hedge_tuning.csv")      # unhedged rule = eta 0
    tune = run([(f, b, n, SEEDS_TUNE) for f in KAPPA for b in BELIEFS for n in N0S], args.workers)
    tune.to_csv(RESULTS / "adaptive_tuning.csv", index=False)
    m = tune.groupby(["feeder", "belief", "n0"])["return_sum"].mean().unstack("n0")
    choice = {}
    for f in KAPPA:
        best_true = hedge_tune[(hedge_tune.feeder == f) & (hedge_tune.belief == "true")
                               & (hedge_tune.eta == 0.0)]["return_sum"].mean()
        regret = best_true - m.loc[f]
        choice[f] = float(regret.max(axis=0).idxmin())
        print(f"\n{f}: mean return by belief and n0\n", m.loc[f].round(4).to_string(),
              f"\nworst-case regret: {regret.max(axis=0).round(4).to_dict()}  chosen n0 = {choice[f]}")
    (RESULTS / "adaptive_choice.json").write_text(json.dumps(choice, indent=2))

    test = run([(f, b, choice[f], SEEDS_TEST) for f in KAPPA for b in BELIEFS], args.workers)
    test.to_csv(RESULTS / "adaptive_evaluation.csv", index=False)
    hedge = pd.read_csv(RESULTS / "hedge_evaluation.csv")
    ppo_true = {"stiff": pd.read_csv(RESULTS / "evaluation_nominal.csv"),
                "weak": pd.read_csv(RESULTS / "hybrid_evaluation.csv")}
    ppo_ce = pd.read_csv(RESULTS / "misspec_trained_evaluation.csv")
    rows = []
    for f in KAPPA:
        for b in BELIEFS:
            x = test[(test.feeder == f) & (test.belief == b)].set_index("scenario_seed")["return_sum"].sort_index()
            hb = hedge[(hedge.feeder == f) & (hedge.belief == b)]
            refs = {"unhedged rule (same belief)": hb[hb.eta == 0.0].set_index("scenario_seed")["return_sum"],
                    "unhedged rule (true model)": hedge[(hedge.feeder == f) & (hedge.belief == "true")
                                                        & (hedge.eta == 0.0)].set_index("scenario_seed")["return_sum"]}
            p = ppo_true[f]
            refs["PPO (true model)"] = p[p.policy.str.match(r"^ppo-s\d$")].groupby("scenario_seed")["return_sum"].mean()
            if b == "class-errors":
                q = ppo_ce[(ppo_ce.feeder == f) & ppo_ce.policy.str.startswith("ppo-s")]
                refs["PPO (believed-trained)"] = q.groupby("scenario_seed")["return_sum"].mean()
            for name, y in refs.items():
                y = y.loc[x.index]
                d = (x - y).values
                if np.allclose(d, 0):
                    continue
                lo, hi = bootstrap_ci(d)
                rows.append({"feeder": f, "belief": b, "n0": choice[f], "reference": name,
                             "adaptive": x.mean(), "ref": y.mean(), "diff": d.mean(),
                             "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue),
                             "wins": int((d > 0).sum()), "rrb": rank_biserial_paired(x.values, y.values)})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "adaptive_stats.csv", index=False)
    pd.set_option("display.width", 230)
    print("\n", st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
