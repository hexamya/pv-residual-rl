"""Hedging the structured rule against acceptance-model errors (follow-up to D26).

HedgedCapacityPolicy mixes the exact expected-capacity allocation with a spread
in proportion to weight x headroom (share eta). eta is chosen on the tuning
scenarios (30000-30049) by minimax regret over five beliefs: the true model and
the four misspecification cases of 28_acceptance_misspec.py. Regret is measured
against the unhedged rule with the true model. The chosen eta is then tested on
the 100 test scenarios on both feeders (stiff: kappa = 0; weak: kappa = 0.5).

Usage: python scripts/30_hedged_rule.py [--workers 4]
Outputs: results/hedge_tuning.csv, hedge_choice.json, hedge_evaluation.csv, hedge_stats.csv
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
ETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
BELIEFS = ["true", "pessimistic", "optimistic", "class-errors", "dispersion"]
KAPPA = {"stiff": 0.0, "weak": 0.5}


def _job(job: tuple) -> pd.DataFrame:
    from pvplan import pvgis
    from pvplan.baselines import HedgedCapacityPolicy
    from pvplan.environment import EnvConfig
    from pvplan.evaluation import evaluate_policy
    feeder, belief, eta, seeds = job
    if belief != "true":
        importlib.import_module("28_acceptance_misspec").install_belief(belief)
    cfg = EnvConfig(impedance_scale=0.5 if feeder == "stiff" else 1.0)
    df = evaluate_policy(f"hedged eta={eta}", HedgedCapacityPolicy(KAPPA[feeder], eta),
                         pvgis.load_summer_days(), cfg, seeds)
    df["feeder"], df["belief"], df["eta"] = feeder, belief, eta
    return df


def run(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(_job, jobs)), ignore_index=True)


def main() -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # ---- tuning: minimax regret over beliefs, per feeder
    tune = run([(f, b, e, SEEDS_TUNE) for f in KAPPA for b in BELIEFS for e in ETAS], args.workers)
    tune.to_csv(RESULTS / "hedge_tuning.csv", index=False)
    m = tune.groupby(["feeder", "belief", "eta"])["return_sum"].mean().unstack("eta")
    choice = {}
    for f in KAPPA:
        best_true = m.loc[(f, "true"), 0.0]
        regret = best_true - m.loc[f]                      # rows: belief, cols: eta
        choice[f] = float(regret.max(axis=0).idxmin())
        print(f"\n{f}: mean return by belief and eta\n", m.loc[f].round(4).to_string(),
              f"\nworst-case regret by eta: {regret.max(axis=0).round(4).to_dict()}",
              f"\nchosen eta = {choice[f]}")
    (RESULTS / "hedge_choice.json").write_text(json.dumps(choice, indent=2))

    # ---- test: chosen eta vs unhedged rule, all beliefs, plus believed-trained PPO
    test = run([(f, b, e, SEEDS_TEST) for f in KAPPA for b in BELIEFS
                for e in sorted({0.0, choice[f]})], args.workers)
    test.to_csv(RESULTS / "hedge_evaluation.csv", index=False)
    ppo_true = {"stiff": pd.read_csv(RESULTS / "evaluation_nominal.csv"),
                "weak": pd.read_csv(RESULTS / "hybrid_evaluation.csv")}
    ppo_ce = pd.read_csv(RESULTS / "misspec_trained_evaluation.csv")
    rows = []
    for f in KAPPA:
        for b in BELIEFS:
            g = test[(test.feeder == f) & (test.belief == b)]
            h = g[g.eta == choice[f]].set_index("scenario_seed")["return_sum"].sort_index()
            refs = {"unhedged rule": g[g.eta == 0.0].set_index("scenario_seed")["return_sum"]}
            p = ppo_true[f]
            refs["PPO (true model)"] = p[p.policy.str.match(r"^ppo-s\d$")].groupby(
                "scenario_seed")["return_sum"].mean()
            if b == "class-errors":
                q = ppo_ce[(ppo_ce.feeder == f) & ppo_ce.policy.str.startswith("ppo-s")]
                refs["PPO (believed-trained)"] = q.groupby("scenario_seed")["return_sum"].mean()
            for name, y in refs.items():
                y = y.loc[h.index]
                d = (h - y).values
                if np.allclose(d, 0):
                    continue
                lo, hi = bootstrap_ci(d)
                rows.append({"feeder": f, "belief": b, "eta": choice[f], "reference": name,
                             "hedged": h.mean(), "ref": y.mean(), "diff": d.mean(),
                             "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": float(sps.wilcoxon(h.values, y.values).pvalue),
                             "wins": int((d > 0).sum()), "rrb": rank_biserial_paired(h.values, y.values)})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "hedge_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n", st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
