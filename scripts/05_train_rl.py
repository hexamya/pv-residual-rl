"""E2 — Train DRL agents (PPO / SAC / tranche-DQN) on the deployment MDP.

Usage:
  python scripts/05_train_rl.py --algo ppo --seed 0 --steps 500000
  python scripts/05_train_rl.py --algo all --seeds 5

Design (proposal 8-4 + corrections D07):
  - PPO/SAC: continuous env, action = softmax logits over [reserve, 32 buses];
  - DQN: tranche env (annual budget in 8 tranches, one bus/reserve per sub-step),
    per-sub-step gamma = 0.98**(1/8) so the *annual* discount matches PPO/SAC;
  - identical MLP trunk [256, 256] for comparability (proposal 8-4-2);
  - EvalCallback on 20 fixed held-out scenarios, model checkpointing, CSV logs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN, PPO, SAC

# Tiny MLPs + per-step gradient updates: multi-threaded CPU GEMM is pure
# overhead (observed 5.5 steps/s for SAC with default threading vs >100 with 1).
torch.set_num_threads(1)
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pvplan import pvgis
from pvplan.environment import EnvConfig, PVDeploymentEnv, TranchePVDeploymentEnv

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"
LOGS = ROOT / "results" / "logs"
ANNUAL_GAMMA = 0.98
N_TRANCHES = 8

pv_days = pvgis.load_summer_days()


def make_env(cfg: EnvConfig, tranche: bool, seed: int):
    def _f():
        if tranche:
            env = TranchePVDeploymentEnv(pv_days, cfg, seed=seed,
                                         n_tranches=N_TRANCHES)
        else:
            env = PVDeploymentEnv(pv_days, cfg, seed=seed)
        return Monitor(env)
    return _f


def train_one(algo: str, seed: int, steps: int, cfg: EnvConfig | None = None,
              tag: str = "nominal") -> Path:
    cfg = cfg or EnvConfig()
    tranche = algo == "dqn"
    n_envs = 16 if algo == "ppo" else 1
    venv = DummyVecEnv([make_env(cfg, tranche, seed * 1000 + i)
                        for i in range(n_envs)])
    eval_env = DummyVecEnv([make_env(cfg, tranche, 999_999)])

    out = MODELS / tag
    out.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS / tag / f"{algo}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(net_arch=[256, 256])
    common = dict(policy="MlpPolicy", env=venv, seed=seed, verbose=0,
                  policy_kwargs=policy_kwargs)
    if algo == "ppo":
        model = PPO(**common, learning_rate=3e-4, n_steps=128, batch_size=256,
                    n_epochs=10, gamma=ANNUAL_GAMMA, gae_lambda=0.95,
                    ent_coef=0.01, clip_range=0.2)
    elif algo == "sac":
        model = SAC(**common, learning_rate=3e-4, buffer_size=100_000,
                    learning_starts=1_000, batch_size=256, tau=0.005,
                    gamma=ANNUAL_GAMMA, train_freq=1, gradient_steps=1)
    elif algo == "dqn":
        model = DQN(**common, learning_rate=1e-4, buffer_size=100_000,
                    learning_starts=2_000, batch_size=256,
                    gamma=ANNUAL_GAMMA ** (1 / N_TRANCHES),
                    exploration_fraction=0.3, exploration_final_eps=0.05,
                    target_update_interval=1_000, train_freq=4)
    else:
        raise ValueError(algo)

    cb = EvalCallback(eval_env, n_eval_episodes=20,
                      eval_freq=max(5_000 // n_envs, 500),
                      log_path=str(log_dir), best_model_save_path=str(log_dir),
                      deterministic=True, verbose=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=steps, callback=cb)
    dt = time.perf_counter() - t0

    path = out / f"{algo}_seed{seed}.zip"
    model.save(path)
    ev = np.load(log_dir / "evaluations.npz")
    pd.DataFrame({
        "timesteps": ev["timesteps"],
        "mean_return": ev["results"].mean(axis=1),
        "std_return": ev["results"].std(axis=1),
    }).to_csv(log_dir / "learning_curve.csv", index=False)
    best = float(ev["results"].mean(axis=1).max())
    print(f"[{tag}] {algo} seed={seed}: {steps} steps in {dt/60:.1f} min, "
          f"best eval return={best:.4f}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac", "dqn", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=0,
                    help="if >0, train seeds 0..seeds-1 (overrides --seed)")
    ap.add_argument("--steps", type=int, default=0,
                    help="0 = per-algo default (ppo 500k, sac 300k, dqn 600k)")
    ap.add_argument("--tag", default="nominal")
    args = ap.parse_args()

    # SAC budget set from observed convergence (~plateau by 90-100k, seed 0).
    defaults = {"ppo": 500_000, "sac": 150_000, "dqn": 600_000}
    algos = ["ppo", "sac", "dqn"] if args.algo == "all" else [args.algo]
    seeds = list(range(args.seeds)) if args.seeds > 0 else [args.seed]
    for algo in algos:
        steps = args.steps or defaults[algo]
        for seed in seeds:
            train_one(algo, seed, steps, tag=args.tag)
