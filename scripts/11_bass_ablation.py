"""E7 — Bass-diffusion adoption ablation (upgrade of proposal limitation).

Environment variant: acceptance mean follows p_i + q*penetration_i (imitation),
so early acceptance is lower and *seeding* a neighborhood raises its future
acceptance. Question: does the RL advantage persist — or grow — when adoption
has diffusion dynamics that reward early, spatially-deliberate seeding?

Compares: PPO trained in the bass env (2 seeds) vs rolling-LP (misspecified:
plans with static expected acceptance) vs peak-first vs random, all evaluated
in the bass env on 100 paired scenarios.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
train_mod = importlib.import_module("05_train_rl")

from stable_baselines3 import PPO

from pvplan import pvgis
from pvplan.baselines import PeakFirstPolicy, RandomPolicy, RollingLPPolicy
from pvplan.environment import EnvConfig
from pvplan.evaluation import evaluate_policy
from pvplan.stats import paired_comparison_table
from pvplan.uncertainty import UncertaintyConfig

RESULTS = ROOT / "results"
SEEDS_EVAL = [10_000 + k for k in range(100)]
cfg = EnvConfig(uncertainty=UncertaintyConfig(adoption_model="bass"))
pv_days = pvgis.load_summer_days()


class Wrap:
    def __init__(self, m):
        self.m = m

    def __call__(self, core, obs):
        return self.m.predict(obs, deterministic=True)[0]


frames = []
for seed in (0, 1):
    path = RESULTS / "models" / "bass" / f"ppo_seed{seed}.zip"
    if not path.exists():
        train_mod.train_one("ppo", seed, 500_000, cfg=cfg, tag="bass")
    df = evaluate_policy(f"ppo-bass-s{seed}", Wrap(PPO.load(path, device="cpu")),
                         pv_days, cfg, SEEDS_EVAL)
    df["algo"] = "ppo-bass"
    frames.append(df)
    print(f"ppo-bass seed {seed}: {df['return_sum'].mean():.4f}", flush=True)

# nominal-trained PPO transferred into the bass env (robustness check)
nom = PPO.load(RESULTS / "models" / "nominal" / "ppo_seed0.zip", device="cpu")
df = evaluate_policy("ppo-nominal-in-bass", Wrap(nom), pv_days, cfg, SEEDS_EVAL)
frames.append(df)

for name, pol in {"rolling-lp": RollingLPPolicy(),
                  "peak-first": PeakFirstPolicy(),
                  "random": RandomPolicy(7)}.items():
    df = evaluate_policy(name, pol, pv_days, cfg, SEEDS_EVAL)
    frames.append(df)
    print(f"{name}: {df['return_sum'].mean():.4f}", flush=True)

all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv(RESULTS / "bass_ablation.csv", index=False)
comp = all_df.copy()
comp.loc[comp.get("algo").notna() if "algo" in comp else [], "policy"] = "ppo-bass"
comp = comp.groupby(["policy", "scenario_seed"]).mean(numeric_only=True).reset_index()

summary = comp.groupby("policy").agg(
    return_mean=("return_sum", "mean"),
    dpeak_final_pct=("dpeak_frac_final", lambda s: 100 * s.mean()),
    installed_mw=("installed_total_mw", "mean"),
).round(4).sort_values("return_mean", ascending=False)
print(summary.to_string())
summary.to_csv(RESULTS / "bass_summary.csv")
tbl = paired_comparison_table(comp, "return_sum", "rolling-lp")
tbl.to_csv(RESULTS / "bass_stats.csv", index=False)
print(tbl[["policy", "mean", "mean_diff", "wilcoxon_p", "cliffs_delta"]]
      .to_string(index=False))
