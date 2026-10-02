"""Retrain the residual PPO in the settings where the frozen policy only tied.

For fairness both sides adapt to each setting: the rule's kappa is re-tuned on
the validation seeds (30000-30049), and the residual PPO is retrained (one
seed, 500k steps, as in the paper's OAT retraining) on top of that re-tuned
rule. Frozen results come from results/robust_evaluation.csv.

Usage:
  python scripts/20_retrain_residual.py tune
  python scripts/20_retrain_residual.py train --setting bess_share=0.2
  python scripts/20_retrain_residual.py eval
Outputs: results/retrain_tuning.csv, retrain_kappa.json, retrain_evaluation.csv,
         retrain_stats.csv
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

from pvplan import pvgis

RESULTS = ROOT / "results"
TIE_SETTINGS = ["bess_share=0.2", "bess_share=0.3", "growth_mean=0.04",
                "horizon_years=5", "annual_budget_mw=0.125", "annual_budget_mw=0.375",
                "bass"]


def _mods():
    return (importlib.import_module("17_weak_feeder"), importlib.import_module("18_hybrid"),
            importlib.import_module("19_hybrid_robustness"))


def tag_for(setting: str) -> str:
    return "weak_residual_" + setting.replace("=", "_")


def _rule_job(job: tuple) -> pd.DataFrame:
    from pvplan.evaluation import evaluate_policy
    weak, _, rob = _mods()
    setting, kappa, seeds, label = job
    df = evaluate_policy(label, weak.make_policy("max-exp-cap", kappa),
                         pvgis.load_summer_days(), rob.settings()[setting], seeds)
    df["setting"], df["param"] = setting, kappa
    return df


def _hybrid_job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    _, hyb, rob = _mods()
    setting, kappa, seeds = job
    pol = hyb.HybridPolicy("residual", 0, tag=tag_for(setting), kappa=kappa)
    df = evaluate_policy("residual-retrained", pol, pvgis.load_summer_days(),
                         rob.settings()[setting], seeds)
    df["setting"] = setting
    return df


def run(fn, jobs: list, workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(fn, jobs)), ignore_index=True)


def cmd_tune(workers: int) -> None:
    weak, _, _ = _mods()
    jobs = [(s, k, weak.SEEDS_TUNE, f"max-exp-cap-V k={k}")
            for s in TIE_SETTINGS for k in weak.KAPPAS]
    df = run(_rule_job, jobs, workers)
    df.to_csv(RESULTS / "retrain_tuning.csv", index=False)
    means = df.groupby(["setting", "param"])["return_sum"].mean()
    best = {s: float(means.loc[s].idxmax()) for s in TIE_SETTINGS}
    (RESULTS / "retrain_kappa.json").write_text(json.dumps(best, indent=2))
    print(means.unstack().round(4).to_string(), "\nselected:", best)


def cmd_train(setting: str, steps: int) -> None:
    _, hyb, rob = _mods()
    kappa = json.loads((RESULTS / "retrain_kappa.json").read_text())[setting]
    hyb.train_one("residual", 0, steps, cfg=rob.settings()[setting], kappa=kappa,
                  tag=tag_for(setting))


def cmd_eval(workers: int) -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci
    _, _, rob = _mods()
    kappas = json.loads((RESULTS / "retrain_kappa.json").read_text())
    new = pd.concat([
        run(_hybrid_job, [(s, kappas[s], rob.SEEDS) for s in TIE_SETTINGS], workers),
        run(_rule_job, [(s, kappas[s], rob.SEEDS, "max-exp-cap-V retuned")
                        for s in TIE_SETTINGS], workers),
    ], ignore_index=True)
    new.to_csv(RESULTS / "retrain_evaluation.csv", index=False)

    frozen = pd.read_csv(RESULTS / "robust_evaluation.csv")
    frozen = frozen[frozen.setting.isin(TIE_SETTINGS)].copy()
    frozen.loc[frozen.policy == "residual-s0", "policy"] = "residual-frozen-s0"
    frozen["family"] = frozen.policy.str.replace(r"-s\d$", "", regex=True)
    frozen = (frozen[frozen.family.isin(["residual", "max-exp-cap-V", "rolling-lp-V"])
                     | (frozen.policy == "residual-frozen-s0")])
    frozen["policy"] = np.where(frozen.policy == "residual-frozen-s0",
                                "residual-frozen-s0", frozen.family)
    frozen.loc[frozen.policy == "residual", "policy"] = "residual-frozen"
    allp = pd.concat([frozen, new], ignore_index=True)
    comp = allp.groupby(["setting", "policy", "scenario_seed"]).mean(
        numeric_only=True).reset_index()

    pairs = [("residual-retrained", "max-exp-cap-V retuned"),
             ("residual-retrained", "max-exp-cap-V"),
             ("residual-retrained", "residual-frozen-s0"),
             ("residual-retrained", "rolling-lp-V"),
             ("max-exp-cap-V retuned", "max-exp-cap-V")]
    rows = []
    for s in TIE_SETTINGS:
        c = comp[comp.setting == s]
        for a, b in pairs:
            x = c[c.policy == a].set_index("scenario_seed")["return_sum"]
            y = c[c.policy == b].set_index("scenario_seed")["return_sum"].loc[x.index]
            d = x.values - y.values
            lo, hi = bootstrap_ci(d)
            rows.append({"setting": s, "policy": a, "reference": b, "diff": d.mean(),
                         "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": 1.0 if np.allclose(d, 0)
                         else float(sps.wilcoxon(x.values, y.values).pvalue),
                         "wins": int((d > 0).sum()), "n": len(d)})
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "retrain_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(comp.pivot_table(index="setting", columns="policy", values="return_sum")
          .loc[TIE_SETTINGS].round(4).to_string())
    print("\nkappa per setting:", kappas)
    print("\n", stats.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "train", "eval"])
    ap.add_argument("--setting", choices=TIE_SETTINGS)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    if args.cmd == "tune":
        cmd_tune(args.workers)
    elif args.cmd == "train":
        cmd_train(args.setting, args.steps)
    else:
        cmd_eval(args.workers)
