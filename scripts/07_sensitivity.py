"""E4 — OAT sensitivity + robustness analysis (decision D13).

Two complementary analyses around the nominal configuration:

  A) ROBUSTNESS (no retraining): the nominal-trained best RL policy and the
     strongest baseline are evaluated in perturbed environments. Answers: "does
     the learned policy survive parameter drift?" — the policy-relevant question.
  B) RETRAINED sensitivity (1 seed per config, best algo): retrain in the
     perturbed environment. Answers: "how much performance was available if the
     perturbation had been known?" (adaptation gap = B - A).

Axes (one-at-a-time from nominal):
  bess_share      {0, 0.1, 0.2, 0.3}          (D14)
  growth_mean     {0.01, 0.02, 0.03, 0.04}
  horizon_years   {5, 10, 15}
  annual_budget   {0.125, 0.25, 0.375} MW
  impedance_scale {0.5, 1.0}   (stiff urban vs weak feeder — topology variant)
  potential_scale {1.0, 1.5, 2.0}

Usage: python scripts/07_sensitivity.py [--retrain] [--algo ppo] [--episodes 50]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pvplan import pvgis
from pvplan.baselines import RollingLPPolicy
from pvplan.environment import EnvConfig
from pvplan.evaluation import evaluate_policy
from pvplan.uncertainty import UncertaintyConfig

RESULTS = ROOT / "results"

AXES: dict[str, list] = {
    "bess_share": [0.0, 0.1, 0.2, 0.3],
    "growth_mean": [0.01, 0.02, 0.03, 0.04],
    "horizon_years": [5, 10, 15],
    "annual_budget_mw": [0.125, 0.25, 0.375],
    "impedance_scale": [0.5, 1.0],
    "potential_scale": [1.0, 1.5, 2.0],
}
NOMINAL = {"bess_share": 0.0, "growth_mean": 0.025, "horizon_years": 10,
           "annual_budget_mw": 0.25, "impedance_scale": 0.5,
           "potential_scale": 1.0}


def build_cfg(axis: str, value) -> EnvConfig:
    kw = dict(NOMINAL)
    kw[axis] = value
    unc = UncertaintyConfig(growth_mean=kw.pop("growth_mean"))
    return EnvConfig(uncertainty=unc, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--best-seed", type=int, default=0,
                    help="nominal-trained seed to use for robustness")
    ap.add_argument("--retrain-steps", type=int, default=300_000)
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC

    pv_days = pvgis.load_summer_days()
    seeds = [40_000 + k for k in range(args.episodes)]
    cls = {"ppo": PPO, "sac": SAC}[args.algo]
    nominal_model = cls.load(
        RESULTS / "models" / "nominal" / f"{args.algo}_seed{args.best_seed}.zip",
        device="cpu")

    class Wrap:
        def __init__(self, m):
            self.m = m

        def __call__(self, core, obs):
            return self.m.predict(obs, deterministic=True)[0]

    rows = []
    for axis, values in AXES.items():
        for v in values:
            cfg = build_cfg(axis, v)
            for pol_name, pol in [(f"{args.algo}-nominal", Wrap(nominal_model)),
                                  ("rolling-lp", RollingLPPolicy())]:
                df = evaluate_policy(pol_name, pol, pv_days, cfg, seeds)
                rows.append({
                    "axis": axis, "value": v, "policy": pol_name,
                    "mode": "robustness",
                    "return_mean": df["return_sum"].mean(),
                    "return_std": df["return_sum"].std(),
                    "dpeak_final_pct": 100 * df["dpeak_frac_final"].mean(),
                    "overload_delta": df["overload_delta_sum"].mean(),
                    "volt_delta": df["volt_delta_sum"].mean(),
                })
                print(f"[robust] {axis}={v} {pol_name}: "
                      f"{rows[-1]['return_mean']:.4f}", flush=True)

            if args.retrain and v != NOMINAL.get(axis):
                sys.path.insert(0, str(ROOT / "scripts"))
                import importlib
                train_mod = importlib.import_module("05_train_rl")
                tag = f"oat_{axis}_{v}"
                path = train_mod.train_one(args.algo, 0, args.retrain_steps,
                                           cfg=cfg, tag=tag)
                model = cls.load(path, device="cpu")
                df = evaluate_policy(f"{args.algo}-retrained", Wrap(model),
                                     pv_days, cfg, seeds)
                rows.append({
                    "axis": axis, "value": v, "policy": f"{args.algo}-retrained",
                    "mode": "retrained",
                    "return_mean": df["return_sum"].mean(),
                    "return_std": df["return_sum"].std(),
                    "dpeak_final_pct": 100 * df["dpeak_frac_final"].mean(),
                    "overload_delta": df["overload_delta_sum"].mean(),
                    "volt_delta": df["volt_delta_sum"].mean(),
                })
                print(f"[retrain] {axis}={v}: {rows[-1]['return_mean']:.4f}",
                      flush=True)

    out = pd.DataFrame(rows)
    suffix = "retrain" if args.retrain else "robust"
    out.to_csv(RESULTS / f"sensitivity_{suffix}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
