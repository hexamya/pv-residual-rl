"""Learning agents trained in a misspecified simulator.

In 28_acceptance_misspec.py only the structured policies plan with a wrong
acceptance model, while plain PPO was trained in the true simulator. In
practice PPO would be trained in a simulator built from the same estimates.
Here PPO and the residual agent are trained in a simulator whose class
acceptance means follow the "class-errors" belief (which reverses the believed
order of residential and commercial). They are then evaluated in the true
environment. The residual agent's allocation layer also plans with the belief.
Three seeds per agent; voltage weight kappa = 0.5 as in the nominal rule.

Usage:
  python scripts/29_misspec_trained.py train --kind ppo-stiff --seed 0
  python scripts/29_misspec_trained.py train --kind ppo-weak --seed 0
  python scripts/29_misspec_trained.py train --kind residual-weak --seed 0
  python scripts/29_misspec_trained.py eval
Outputs: results/misspec_trained_evaluation.csv, misspec_trained_stats.csv
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pvplan.environment import EnvConfig
from pvplan.uncertainty import UncertaintyConfig

RESULTS = ROOT / "results"
SEEDS = [10_000 + k for k in range(100)]
CE = {"residential": 1.25, "mixed": 0.85, "commercial": 0.80, "admin": 1.15}
IMP = {"stiff": 0.5, "weak": 1.0}
N_SEEDS = 3


def cfg(feeder: str, believed: bool) -> EnvConfig:
    unc = UncertaintyConfig(accept_class_scale=CE) if believed else UncertaintyConfig()
    return EnvConfig(impedance_scale=IMP[feeder], uncertainty=unc)


def cmd_train(kind: str, seed: int, steps: int) -> None:
    algo, feeder = kind.split("-")
    if algo == "ppo":
        importlib.import_module("05_train_rl").train_one(
            "ppo", seed, steps, cfg=cfg(feeder, True), tag=f"misspec_ppo_{feeder}")
    else:
        importlib.import_module("18_hybrid").train_one(
            "residual", seed, steps, cfg=cfg("weak", True), kappa=0.5,
            tag="misspec_residual_weak")


def _job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan import pvgis
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    feeder, label = job
    weak = importlib.import_module("17_weak_feeder")
    fam, _, seed = label.rpartition("-s")
    if fam == "ppo":
        pol = weak.SB3Policy(RESULTS / "logs" / f"misspec_ppo_{feeder}" / f"ppo_seed{seed}" / "best_model.zip")
    else:
        importlib.import_module("28_acceptance_misspec").install_belief("class-errors")
        pol = importlib.import_module("18_hybrid").HybridPolicy(
            "residual", int(seed), tag="misspec_residual_weak", kappa=0.5)
    df = evaluate_policy(label, pol, pvgis.load_summer_days(), cfg(feeder, False), SEEDS)
    df["feeder"] = feeder
    print(f"  {feeder} {label}: {df['return_sum'].mean():.4f}", flush=True)
    return df


def cmd_eval(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired
    jobs = ([("stiff", f"ppo-s{s}") for s in range(N_SEEDS)]
            + [("weak", f"ppo-s{s}") for s in range(N_SEEDS)]
            + [("weak", f"residual-s{s}") for s in range(N_SEEDS)])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        df = pd.concat(list(ex.map(_job, jobs)), ignore_index=True)
    df.to_csv(RESULTS / "misspec_trained_evaluation.csv", index=False)

    fam = lambda d: d.assign(family=d.policy.str.replace(r"-s\d$", "", regex=True)).groupby(
        ["family", "scenario_seed"])["return_sum"].mean()
    mis = pd.read_csv(RESULTS / "misspec_evaluation.csv")
    mis = mis[mis.case == "class-errors"]
    rule = {f: fam(mis[mis.feeder == f]) for f in ("stiff", "weak")}
    true_ref = {"stiff": fam(pd.read_csv(RESULTS / "evaluation_nominal.csv")),
                "weak": fam(pd.read_csv(RESULTS / "hybrid_evaluation.csv"))}
    rows = []
    for feeder, g in df.groupby("feeder"):
        tr = fam(g)
        comps = [("ppo", rule[feeder], "max-exp-cap" if feeder == "stiff" else "max-exp-cap-V",
                  "rule (believed)"),
                 ("ppo", true_ref[feeder], "ppo", "PPO trained on the true model")]
        if feeder == "weak":
            comps += [("residual", rule[feeder], "max-exp-cap-V", "rule (believed)"),
                      ("residual", tr, "ppo", "PPO (believed-trained)"),
                      ("residual", true_ref[feeder], "residual", "residual trained on the true model")]
        for a, refser, b, bname in comps:
            x = tr.loc[a]
            y = refser.loc[b].loc[x.index]
            d = (x - y).values
            lo, hi = bootstrap_ci(d)
            rows.append({"feeder": feeder, "policy": f"{a} (believed-trained)", "reference": bname,
                         "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue),
                         "wins": int((d > 0).sum()), "rrb": rank_biserial_paired(x.values, y.values)})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "misspec_trained_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--kind", choices=["ppo-stiff", "ppo-weak", "residual-weak"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cmd_train(a.kind, a.seed, a.steps) if a.cmd == "train" else cmd_eval(a.workers)
