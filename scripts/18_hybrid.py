"""Hybrid design on the weak feeder: PPO sets priorities, an analytic layer allocates.

Variants (HybridPVDeploymentEnv, logit scale 2):
  hybrid   — weights w_i = exp(2 u_i): the agent must learn placement itself;
  residual — weights w_i = (1 + kappa* s_i) exp(2 u_i): the agent corrects the
             tuned max-exp-cap-V rule (u = 0 reproduces the rule exactly).
PPO hyperparameters, seeds, validation stream and checkpoint selection are
identical to scripts/05_train_rl.py.

Usage:
  python scripts/18_hybrid.py train --variant hybrid --seed 0
  python scripts/18_hybrid.py eval
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pvplan import pvgis
from pvplan.environment import HybridPVDeploymentEnv, PlanningCore, hybrid_offers
from pvplan.allocation import voltage_relief_weights

weak = importlib.import_module("17_weak_feeder")
RESULTS = ROOT / "results"
VARIANTS = ("hybrid", "residual")


def prior_kappa(variant: str) -> float:
    if variant == "hybrid":
        return 0.0
    return float(json.loads((RESULTS / "weak_tuned.json").read_text())["kappa"])


def train_one(variant: str, seed: int, steps: int, cfg=None,
              kappa: float | None = None, tag: str | None = None) -> Path:
    """Train one seed; cfg/kappa/tag default to the nominal weak-feeder run."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)

    pv_days = pvgis.load_summer_days()
    cfg = cfg or weak.weak_cfg()
    kappa = prior_kappa(variant) if kappa is None else kappa
    tag = tag or f"weak_{variant}"

    def make(env_seed: int):
        return lambda: Monitor(HybridPVDeploymentEnv(
            pv_days, cfg, seed=env_seed, prior_kappa=kappa))

    venv = DummyVecEnv([make(seed * 1000 + i) for i in range(16)])
    eval_env = DummyVecEnv([make(999_999)])
    out = RESULTS / "models" / tag
    out.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS / "logs" / tag / f"ppo_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=3e-4, n_steps=128, batch_size=256, n_epochs=10,
                gamma=0.98, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2)
    cb = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=max(5_000 // 16, 500),
                      log_path=str(log_dir), best_model_save_path=str(log_dir),
                      deterministic=True, verbose=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=cb)
    path = out / f"ppo_seed{seed}.zip"
    model.save(path)
    ev = np.load(log_dir / "evaluations.npz")
    print(f"[{tag}] seed={seed} kappa={kappa}: {steps} steps in "
          f"{(time.perf_counter() - t0) / 60:.1f} min, best eval return="
          f"{ev['results'].mean(axis=1).max():.4f}", flush=True)
    return path


class HybridPolicy:
    """Runs a trained hybrid agent inside the standard PVDeploymentEnv."""

    def __init__(self, variant: str, seed: int, tag: str | None = None,
                 kappa: float | None = None):
        from stable_baselines3 import PPO
        from pvplan.baselines import _offered_to_action
        self._to_action = _offered_to_action
        tag = tag or f"weak_{variant}"
        best = RESULTS / "logs" / tag / f"ppo_seed{seed}" / "best_model.zip"
        self.model = PPO.load(best if best.exists()
                              else RESULTS / "models" / tag / f"ppo_seed{seed}.zip",
                              device="cpu")
        self.kappa = prior_kappa(variant) if kappa is None else kappa

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        s = voltage_relief_weights(core)
        u = self.model.predict(np.concatenate([obs, s]).astype(np.float32),
                               deterministic=True)[0]
        offered = hybrid_offers(core, u, s, self.kappa, 2.0)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        return self._to_action(offered, budget)


def _eval_job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    label, kind, arg = job
    pol = (HybridPolicy(kind, arg) if kind in VARIANTS
           else weak.make_policy("max-exp-cap", arg))
    df = evaluate_policy(label, pol, pvgis.load_summer_days(), weak.weak_cfg(),
                         weak.SEEDS_TEST)
    print(f"  {label}: {df['return_sum'].mean():.4f}", flush=True)
    return df


def cmd_eval(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci

    kappa = prior_kappa("residual")
    jobs = [(f"{v}-s{s}", v, s) for v in VARIANTS for s in range(5)]
    jobs += [("max-exp-cap", "rule", 0.0), ("max-exp-cap-V", "rule", kappa)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        new = pd.concat(list(ex.map(_eval_job, jobs)), ignore_index=True)
    old = pd.read_csv(RESULTS / "weak_evaluation.csv")
    old = old[~old.policy.str.startswith("max-exp-cap")]   # superseded (exact layer)
    df = pd.concat([old, new], ignore_index=True)
    df.to_csv(RESULTS / "hybrid_evaluation.csv", index=False)

    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    per_seed = (df[df.policy.str.match(r".*-s\d$")]
                .groupby(["family", "policy"])["return_sum"].mean())
    comp = df.groupby(["family", "scenario_seed"]).mean(numeric_only=True).reset_index()
    comp["v_term"] = -0.5 * comp["volt_delta_sum"]
    comp["peak_term"] = comp["return_sum"] - comp["v_term"] + 0.5 * comp["overload_delta_sum"]
    summary = comp.groupby("family").agg(
        return_mean=("return_sum", "mean"), peak_term=("peak_term", "mean"),
        v_term=("v_term", "mean"), installed_mw=("installed_total_mw", "mean"),
    ).sort_values("return_mean", ascending=False)
    summary.to_csv(RESULTS / "hybrid_summary.csv")

    rows = []
    for fam in ["residual", "hybrid", "ppo"]:
        for ref in ["max-exp-cap-V", "rolling-lp-V", "ppo", "hybrid"]:
            if ref == fam:
                continue
            x = comp[comp.family == fam].set_index("scenario_seed")["return_sum"]
            y = comp[comp.family == ref].set_index("scenario_seed")["return_sum"].loc[x.index]
            d = x.values - y.values
            lo, hi = bootstrap_ci(d)
            rows.append({"policy": fam, "reference": ref, "mean": x.mean(),
                         "ref": y.mean(), "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": float(sps.wilcoxon(x.values, y.values).pvalue),
                         "wins": int((d > 0).sum()), "n": len(d)})
    stats = pd.DataFrame(rows)
    stats.to_csv(RESULTS / "hybrid_stats.csv", index=False)
    print("\nper seed:\n", per_seed.round(4).to_string())
    print("\n", summary.round(4).to_string())
    print("\n", stats.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--variant", choices=VARIANTS, default="hybrid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    if args.cmd == "train":
        train_one(args.variant, args.seed, args.steps)
    else:
        cmd_eval(args.workers)
