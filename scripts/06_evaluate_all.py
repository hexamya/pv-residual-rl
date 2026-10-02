"""E2+E3 — Final evaluation: RL agents (all seeds) vs baselines vs oracle.

100 held-out paired scenarios; Wilcoxon signed-rank + Cliff's delta + CIs;
optimality gap vs the perfect-information oracle.

Usage: python scripts/06_evaluate_all.py [n_episodes] [tag]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO, SAC

from pvplan import pvgis
from pvplan.baselines import (CFPolicy, OraclePolicy, PeakFirstPolicy,
                              RandomPolicy, RollingLPPolicy)
from pvplan.environment import EnvConfig
from pvplan.evaluation import evaluate_policy
from pvplan.stats import paired_comparison_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TAG = sys.argv[2] if len(sys.argv) > 2 else "nominal"
SEEDS = [10_000 + k for k in range(N_EPISODES)]

pv_days = pvgis.load_summer_days()
cfg = EnvConfig()


class SB3Continuous:
    def __init__(self, model):
        self.model = model

    def __call__(self, core, obs):
        return self.model.predict(obs, deterministic=True)[0]


class SB3Discrete:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs):
        return int(self.model.predict(obs, deterministic=True)[0])


frames = []

# --- baselines + oracle ---------------------------------------------------
for name, pol in {
    "random": RandomPolicy(seed=7),
    "peak-first": PeakFirstPolicy(),
    "cf-static": CFPolicy(static=True),
    "cf-adaptive": CFPolicy(static=False),
    "rolling-lp": RollingLPPolicy(),
    "oracle": OraclePolicy(),
}.items():
    df = evaluate_policy(name, pol, pv_days, cfg, SEEDS)
    frames.append(df)
    print(f"{name:14s} return={df['return_sum'].mean():.4f}", flush=True)

# --- RL agents (every trained seed) ---------------------------------------
model_dir = RESULTS / "models" / TAG
loaders = {"ppo": PPO, "sac": SAC, "dqn": DQN}
for algo, cls in loaders.items():
    for path in sorted(model_dir.glob(f"{algo}_seed*.zip")):
        seed = path.stem.split("seed")[1]
        # model selection by validation performance (EvalCallback best
        # checkpoint), applied identically to all algorithms — final-step
        # weights can be far off-peak for value-based methods (DQN).
        best = RESULTS / "logs" / TAG / f"{algo}_seed{seed}" / "best_model.zip"
        model = cls.load(best if best.exists() else path, device="cpu")
        if algo == "dqn":
            df = evaluate_policy(f"{algo}-s{seed}", SB3Discrete(model), pv_days,
                                 cfg, SEEDS, tranche=True)
        else:
            df = evaluate_policy(f"{algo}-s{seed}", SB3Continuous(model),
                                 pv_days, cfg, SEEDS)
        df["algo"] = algo
        frames.append(df)
        print(f"{algo}-s{seed:3s}      return={df['return_sum'].mean():.4f}",
              flush=True)

all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv(RESULTS / f"evaluation_{TAG}.csv", index=False)

# per-algorithm aggregate rows (mean over that algo's seeds, per scenario)
rl = all_df[all_df["algo"].notna()] if "algo" in all_df else all_df.iloc[0:0]
agg_frames = [all_df[all_df.get("algo").isna()] if "algo" in all_df else all_df]
for algo, g in rl.groupby("algo"):
    agg = g.groupby("scenario_seed").mean(numeric_only=True).reset_index()
    agg["policy"] = algo
    agg_frames.append(agg)
comp_df = pd.concat(agg_frames, ignore_index=True)

# summary + paired statistics
summary = comp_df.groupby("policy").agg(
    return_mean=("return_sum", "mean"),
    return_std=("return_sum", "std"),
    dpeak_final_pct=("dpeak_frac_final", lambda s: 100 * s.mean()),
    dpeak_mean_pct=("dpeak_frac_mean", lambda s: 100 * s.mean()),
    overload_delta=("overload_delta_sum", "mean"),
    volt_delta=("volt_delta_sum", "mean"),
    installed_mw=("installed_total_mw", "mean"),
    budget_eff=("budget_efficiency", "mean"),
).round(4).sort_values("return_mean", ascending=False)
summary.to_csv(RESULTS / f"summary_{TAG}.csv")
print("\n=== SUMMARY ===\n", summary.to_string())

oracle_mean = comp_df[comp_df.policy == "oracle"]["return_sum"].mean()
summary["oracle_gap_pct"] = (100 * (oracle_mean - summary["return_mean"])
                             / oracle_mean).round(2)
print("\nOptimality gap vs oracle (%):\n",
      summary["oracle_gap_pct"].to_string())

for ref in ["rolling-lp", "peak-first"]:
    tbl = paired_comparison_table(comp_df, "return_sum", ref)
    tbl.to_csv(RESULTS / f"stats_vs_{ref}_{TAG}.csv", index=False)
    print(f"\n=== paired vs {ref} (return_sum) ===")
    print(tbl[["policy", "mean", "mean_diff", "diff_ci_lo", "diff_ci_hi",
               "wilcoxon_p", "cliffs_delta"]].to_string(index=False))

summary.to_csv(RESULTS / f"summary_{TAG}.csv")
