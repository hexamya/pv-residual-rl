"""Path A pilot: the weak feeder (original IEEE 33-bus impedances) as main case.

On the weak feeder, midday PV relieves undervoltage, and the relief depends on
where PV is installed, so placement has value beyond total capacity. This
script trains PPO there and benchmarks it against network-blind and
network-aware rules and a rolling LP with a voltage term.

Usage:
  python scripts/17_weak_feeder.py train --seed 0     # one PPO seed (500k steps)
  python scripts/17_weak_feeder.py diag               # placement-value diagnostics
  python scripts/17_weak_feeder.py tune               # tune baseline weights on validation seeds
  python scripts/17_weak_feeder.py eval               # 100 paired test scenarios
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
from pvplan.environment import EnvConfig

RESULTS = ROOT / "results"
TAG = "weak"
SEEDS_TEST = [10_000 + k for k in range(100)]
SEEDS_TUNE = [30_000 + k for k in range(50)]   # disjoint from test/OAT/training


def weak_cfg() -> EnvConfig:
    return EnvConfig(impedance_scale=1.0)


KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0]  # max-exp-cap-V weight
LAMBDAS = [0.0, 0.001, 0.003, 0.005, 0.01, 0.03, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]  # LP-V weight


class SB3Policy:
    def __init__(self, path: Path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def __call__(self, core, obs):
        return self.model.predict(obs, deterministic=True)[0]


def make_policy(name: str, param: float | None = None):
    from pvplan.baselines import (AcceptGreedyPolicy, MaxExpectedCapacityPolicy,
                                  PeakFirstPolicy, RandomPolicy, RollingLPPolicy,
                                  RollingLPVPolicy)
    if name == "max-exp-cap":
        return MaxExpectedCapacityPolicy(kappa=param or 0.0)
    if name == "rolling-lp-v":
        return RollingLPVPolicy(lam_v=param or 0.0)
    if name.startswith("ppo-s"):
        seed = name.split("-s")[1]
        best = RESULTS / "logs" / TAG / f"ppo_seed{seed}" / "best_model.zip"
        return SB3Policy(best if best.exists() else RESULTS / "models" / TAG / f"ppo_seed{seed}.zip")
    if name == "ppo-nominal":                       # trained on the stiff feeder
        return SB3Policy(RESULTS / "models" / "nominal" / "ppo_seed0.zip")
    return {"accept-greedy": AcceptGreedyPolicy, "peak-first": PeakFirstPolicy,
            "random": lambda: RandomPolicy(seed=7),
            "rolling-lp": RollingLPPolicy}[name]()


def _job(args: tuple) -> pd.DataFrame:
    import torch
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    label, name, param, seeds = args
    df = evaluate_policy(label, make_policy(name, param), pvgis.load_summer_days(),
                         weak_cfg(), seeds)
    df["param"] = param
    print(f"  {label}: {df['return_sum'].mean():.4f}", flush=True)
    return df


def run_jobs(jobs: list[tuple], workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(_job, jobs)), ignore_index=True)


def cmd_tune(workers: int, kappas: list[float], lambdas: list[float],
             merge: bool) -> None:
    """Grid-search the baseline weights; with merge, add to the existing grid."""
    jobs = ([(f"max-exp-cap-V k={k}", "max-exp-cap", k, SEEDS_TUNE) for k in kappas]
            + [(f"rolling-lp-v lam={l}", "rolling-lp-v", l, SEEDS_TUNE) for l in lambdas])
    df = run_jobs(jobs, workers)
    path = RESULTS / "weak_tuning.csv"
    if merge and path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old.policy.isin(df.policy.unique())], df], ignore_index=True)
    df.to_csv(path, index=False)
    means = df.groupby(["policy", "param"])["return_sum"].mean()
    best = {
        "kappa": float(means[means.index.get_level_values(0).str.startswith("max-exp")]
                       .idxmax()[1]),
        "lam_v": float(means[means.index.get_level_values(0).str.startswith("rolling")]
                       .idxmax()[1]),
    }
    with open(RESULTS / "weak_tuned.json", "w") as f:
        json.dump(best, f, indent=2)
    print(means.round(4).to_string(), "\nselected:", best)


def cmd_eval(workers: int) -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci

    best = json.loads((RESULTS / "weak_tuned.json").read_text())
    jobs = [(f"ppo-s{s}", f"ppo-s{s}", None, SEEDS_TEST) for s in range(5)]
    jobs += [("ppo-nominal", "ppo-nominal", None, SEEDS_TEST),
             ("max-exp-cap", "max-exp-cap", 0.0, SEEDS_TEST),
             ("max-exp-cap-V", "max-exp-cap", best["kappa"], SEEDS_TEST),
             ("rolling-lp-V", "rolling-lp-v", best["lam_v"], SEEDS_TEST)]
    jobs += [(n, n, None, SEEDS_TEST)
             for n in ["accept-greedy", "peak-first", "random", "rolling-lp"]]
    df = run_jobs(jobs, workers)
    df.to_csv(RESULTS / "weak_evaluation.csv", index=False)

    ppo = (df[df.policy.str.startswith("ppo-s")].groupby("scenario_seed")
           .mean(numeric_only=True).reset_index())
    ppo["policy"] = "ppo"
    comp = pd.concat([df[~df.policy.str.startswith("ppo-s")], ppo], ignore_index=True)
    comp["peak_term"] = comp["return_sum"] + 0.5 * comp["volt_delta_sum"] \
        + 0.5 * comp["overload_delta_sum"]
    comp["v_term"] = -0.5 * comp["volt_delta_sum"]
    summary = comp.groupby("policy").agg(
        return_mean=("return_sum", "mean"), peak_term=("peak_term", "mean"),
        v_term=("v_term", "mean"), installed_mw=("installed_total_mw", "mean"),
    ).sort_values("return_mean", ascending=False)
    seeds_ret = df[df.policy.str.startswith("ppo-s")].groupby("policy")["return_sum"].mean()
    summary.to_csv(RESULTS / "weak_summary.csv")

    rows = []
    for ref in ["max-exp-cap-V", "rolling-lp-V", "max-exp-cap", "rolling-lp"]:
        y = comp[comp.policy == ref].set_index("scenario_seed")["return_sum"]
        x = ppo.set_index("scenario_seed")["return_sum"].loc[y.index]
        d = x.values - y.values
        lo, hi = bootstrap_ci(d)
        rows.append({"reference": ref, "ppo": x.mean(), "ref": y.mean(),
                     "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                     "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue),
                     "ppo_wins": int((d > 0).sum()), "n": len(d)})
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "weak_stats_ppo.csv", index=False)
    print("\nPPO seeds:", seeds_ret.round(4).to_dict())
    print("\n", summary.round(4).to_string())
    print("\nPPO (seed-averaged) minus reference:\n", stats.round(4).to_string(index=False))


def cmd_train(seed: int, steps: int) -> None:
    train_mod = importlib.import_module("05_train_rl")
    train_mod.train_one("ppo", seed, steps, cfg=weak_cfg(), tag=TAG)


def _random_feasible(rng: np.random.Generator, pot: np.ndarray, total: float) -> np.ndarray:
    x, rem = np.zeros(len(pot)), total
    for i in rng.permutation(len(pot)):
        x[i] = min(pot[i], rem * rng.uniform())
        rem -= x[i]
    for i in rng.permutation(len(pot)):
        add = min(pot[i] - x[i], rem)
        x[i], rem = x[i] + add, rem - add
    return x


def _ordered_fill(order: np.ndarray, pot: np.ndarray, total: float) -> np.ndarray:
    x, rem = np.zeros(len(pot)), total
    for i in order:
        x[i] = min(pot[i], rem)
        rem -= x[i]
    return x


def cmd_diag(pv_days: np.ndarray) -> None:
    from pvplan.environment import N_BUS, PlanningCore
    from pvplan.network import V_MIN_PU

    core = PlanningCore(pv_days, weak_cfg(), seed=0)
    rc_diag = np.diag(core.net.r_common_pu)
    out: dict = {}

    # undervoltage on the noise-free nominal day without PV, by hour
    for year, scale in [(0, 1.0), (5, 1.025 ** 5), (10, 1.025 ** 10)]:
        p = core.p_nom_kw[:, None] * core.shapes * scale
        q = core.q_nom_kvar[:, None] * core.shapes * scale
        v = core.net.voltages_pu(p, q)
        out[f"year{year}_no_pv"] = {
            "vmin_pu": float(v.min()),
            "share_bus_hours_below_0.95": float((v < V_MIN_PU).mean()),
            "violating_hours": [int(h) for h in np.where((v < V_MIN_PU).any(axis=0))[0]],
            "buses_violating_at_13h": int((v[:, 13] < V_MIN_PU).sum()),
        }

    # same total MW, different placement: one year, identical draws
    rng = np.random.default_rng(1)

    def year_terms(x: np.ndarray) -> tuple[float, float]:
        core.reseed(10_000)
        core.reset_state()
        core.load_scale = 1.025 ** 5            # mid-programme demand level
        core.installed_mw = x
        info = core.apply_year(np.zeros(N_BUS), 0.0)
        return info["dpeak_frac"], -0.5 * info["volt_delta"]

    for total in (0.5, 1.0, 1.4):
        rnd = [year_terms(_random_feasible(rng, core.pot_mw, total)) for _ in range(200)]
        far = year_terms(_ordered_fill(np.argsort(-rc_diag), core.pot_mw, total))
        near = year_terms(_ordered_fill(np.argsort(rc_diag), core.pot_mw, total))
        out[f"placement_{total}MW"] = {
            "peak_term_range_random": float(np.ptp([r[0] for r in rnd])),
            "volt_term_random_min": float(min(r[1] for r in rnd)),
            "volt_term_random_max": float(max(r[1] for r in rnd)),
            "volt_term_far_end_first": far[1],
            "volt_term_substation_first": near[1],
            "peak_term": far[0],
        }
    with open(RESULTS / "weak_diag.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "diag", "tune", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--kappas", type=float, nargs="*", default=None,
                    help="tune only these kappas and merge with the existing grid")
    ap.add_argument("--lambdas", type=float, nargs="*", default=None,
                    help="tune only these lambdas and merge with the existing grid")
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args.seed, args.steps)
    elif args.cmd == "diag":
        cmd_diag(pvgis.load_summer_days())
    elif args.cmd == "tune":
        partial = args.kappas is not None or args.lambdas is not None
        cmd_tune(args.workers,
                 args.kappas if args.kappas is not None else ([] if partial else KAPPAS),
                 args.lambdas if args.lambdas is not None else ([] if partial else LAMBDAS),
                 merge=partial)
    elif args.cmd == "eval":
        cmd_eval(args.workers)
