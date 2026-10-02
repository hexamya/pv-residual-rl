"""Does the advantage of the structured layer rely on lost offers?

In the base model, offered capacity that is not accepted (or exceeds rooftop
headroom) is lost from the budget. In this variant it returns to the reserve
(EnvConfig.refund_unaccepted). The budget is lowered to 0.165 MW/yr so that the
voltage-weighted rule installs about as much capacity as in the base case
(about 1.47 MW), which keeps scarcity comparable. The rule's kappa is re-tuned,
and plain PPO and the residual agent are retrained (five seeds each).

Usage:
  python scripts/26_refund_variant.py tune
  python scripts/26_refund_variant.py train --algo ppo --seed 0      (or --algo residual)
  python scripts/26_refund_variant.py eval
Outputs: results/refund_tuning.csv, refund_kappa.json, refund_evaluation.csv, refund_stats.csv
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

RESULTS = ROOT / "results"
BUDGET = 0.165


def cfg_refund() -> EnvConfig:
    return EnvConfig(impedance_scale=1.0, refund_unaccepted=True, annual_budget_mw=BUDGET)


def _mods():
    return importlib.import_module("17_weak_feeder"), importlib.import_module("18_hybrid")


def make(label: str):
    weak, hyb = _mods()
    kappa = json.loads((RESULTS / "refund_kappa.json").read_text())["kappa"]
    fam, _, seed = label.rpartition("-s")
    if fam == "ppo":                                   # trained in the refund variant
        best = RESULTS / "logs" / "weak_refund" / f"ppo_seed{seed}" / "best_model.zip"
        return weak.SB3Policy(best)
    if fam == "ppo-orig":                              # trained on the base weak feeder
        return weak.make_policy(f"ppo-s{seed}")
    if fam == "residual":
        return hyb.HybridPolicy("residual", int(seed), tag="weak_refund_residual", kappa=kappa)
    if fam == "residual-orig":
        return hyb.HybridPolicy("residual", int(seed))
    if label == "max-exp-cap-V":
        return weak.make_policy("max-exp-cap", kappa)
    if label == "max-exp-cap":
        return weak.make_policy("max-exp-cap", 0.0)
    if label == "rolling-lp-V":
        return weak.make_policy("rolling-lp-v", 0.01)
    raise ValueError(label)


def _job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    label, seeds, policy = job
    pol = policy if policy is not None else make(label)
    df = evaluate_policy(label, pol, pvgis.load_summer_days(), cfg_refund(), seeds)
    print(f"  {label}: {df['return_sum'].mean():.4f}", flush=True)
    return df


def run(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(_job, jobs)), ignore_index=True)


def cmd_tune(workers: int) -> None:
    weak, _ = _mods()
    from pvplan.baselines import MaxExpectedCapacityPolicy
    jobs = [(f"max-exp-cap-V k={k}", weak.SEEDS_TUNE, MaxExpectedCapacityPolicy(kappa=k))
            for k in weak.KAPPAS]
    df = run(jobs, workers)
    df.to_csv(RESULTS / "refund_tuning.csv", index=False)
    means = df.groupby("policy")["return_sum"].mean()
    k = float(means.idxmax().split("=")[1])
    (RESULTS / "refund_kappa.json").write_text(json.dumps({"kappa": k, "budget": BUDGET}, indent=2))
    print(means.round(4).to_string(), "\nselected kappa:", k)


def cmd_train(algo: str, seed: int, steps: int) -> None:
    weak, hyb = _mods()
    if algo == "ppo":
        importlib.import_module("05_train_rl").train_one("ppo", seed, steps, cfg=cfg_refund(),
                                                        tag="weak_refund")
    else:
        kappa = json.loads((RESULTS / "refund_kappa.json").read_text())["kappa"]
        hyb.train_one("residual", seed, steps, cfg=cfg_refund(), kappa=kappa,
                      tag="weak_refund_residual")


def cmd_eval(workers: int) -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired
    weak, _ = _mods()
    labels = (["rolling-lp-V", "max-exp-cap-V", "max-exp-cap"]
              + [f"{f}-s{s}" for f in ("ppo", "residual", "ppo-orig", "residual-orig")
                 for s in range(5)])
    df = run([(l, weak.SEEDS_TEST, None) for l in labels], workers)
    df.to_csv(RESULTS / "refund_evaluation.csv", index=False)
    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    comp = df.groupby(["family", "scenario_seed"]).mean(numeric_only=True).reset_index()
    rows = []
    for a, b in [("residual", "max-exp-cap-V"), ("ppo", "max-exp-cap-V"), ("residual", "ppo"),
                 ("residual-orig", "max-exp-cap-V"), ("ppo-orig", "max-exp-cap-V"),
                 ("max-exp-cap-V", "max-exp-cap"), ("residual", "rolling-lp-V")]:
        x = comp[comp.family == a].set_index("scenario_seed")["return_sum"].sort_index()
        y = comp[comp.family == b].set_index("scenario_seed")["return_sum"].loc[x.index]
        d = (x - y).values
        lo, hi = bootstrap_ci(d)
        rows.append({"policy": a, "reference": b, "mean": x.mean(), "ref": y.mean(),
                     "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue),
                     "wins": int((d > 0).sum()), "rrb": rank_biserial_paired(x.values, y.values)})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "refund_stats.csv", index=False)
    summ = comp.groupby("family")[["return_sum", "installed_total_mw", "offered_total_mw"]].mean()
    print(summ.sort_values("return_sum", ascending=False).round(4).to_string())
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "train", "eval"])
    ap.add_argument("--algo", choices=["ppo", "residual"], default="ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    {"tune": lambda: cmd_tune(a.workers), "train": lambda: cmd_train(a.algo, a.seed, a.steps),
     "eval": lambda: cmd_eval(a.workers)}[a.cmd]()
