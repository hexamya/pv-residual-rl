"""E4b — Paired statistics for the OAT robustness / adaptation analysis.

07_sensitivity.py stored only per-configuration means. This script re-runs the
same protocol (50 held-out scenarios, seeds 40000-40049, nominal-trained PPO
seed 0, rolling LP re-solved in each perturbed environment, and the PPO model
retrained in that environment where available) and stores per-scenario results
so that paired differences can be reported with bootstrap CIs, Wilcoxon
signed-rank p-values and Cliff's delta.

Usage: python scripts/15_sensitivity_stats.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sens = importlib.import_module("07_sensitivity")

from stable_baselines3 import PPO  # noqa: E402

from pvplan import pvgis  # noqa: E402
from pvplan.baselines import RollingLPPolicy  # noqa: E402
from pvplan.evaluation import evaluate_policy  # noqa: E402
from pvplan.stats import bootstrap_ci, cliffs_delta  # noqa: E402

RESULTS = ROOT / "results"
SEEDS = [40_000 + k for k in range(50)]


class Wrap:
    def __init__(self, m):
        self.m = m

    def __call__(self, core, obs):
        return self.m.predict(obs, deterministic=True)[0]


def main() -> None:
    pv_days = pvgis.load_summer_days()
    nominal = PPO.load(RESULTS / "models" / "nominal" / "ppo_seed0.zip", device="cpu")
    configs = [("nominal", None)]
    for axis, values in sens.AXES.items():
        for v in values:
            if v != sens.NOMINAL.get(axis):
                configs.append((axis, v))

    frames = []
    for axis, v in configs:
        cfg = sens.build_cfg("bess_share", 0.0) if axis == "nominal" else \
            sens.build_cfg(axis, v)
        pols = [("ppo-frozen", Wrap(nominal)), ("rolling-lp", RollingLPPolicy())]
        rt = RESULTS / "models" / f"oat_{axis}_{v}" / "ppo_seed0.zip"
        if axis != "nominal" and rt.exists():
            pols.append(("ppo-retrained", Wrap(PPO.load(rt, device="cpu"))))
        for name, pol in pols:
            df = evaluate_policy(name, pol, pv_days, cfg, SEEDS)
            df["axis"], df["value"] = axis, v
            frames.append(df)
        print(f"{axis}={v} done", flush=True)

    per = pd.concat(frames, ignore_index=True)
    per.to_csv(RESULTS / "sensitivity_paired.csv", index=False)

    rows = []
    for (axis, v), g in per.groupby(["axis", "value"], dropna=False, sort=False):
        lp = g[g.policy == "rolling-lp"].set_index("scenario_seed")["return_sum"]
        for pol in ("ppo-frozen", "ppo-retrained"):
            x = g[g.policy == pol].set_index("scenario_seed")["return_sum"]
            if x.empty:
                continue
            a, b = x.loc[lp.index].values, lp.values
            d = a - b
            lo, hi = bootstrap_ci(d)
            rows.append({
                "axis": axis, "value": v, "policy": pol,
                "return_mean": a.mean(), "lp_mean": b.mean(),
                "diff_mean": d.mean(), "ci_lo": lo, "ci_hi": hi,
                "wilcoxon_p": float(sps.wilcoxon(a, b).pvalue) if not np.allclose(d, 0) else 1.0,
                "cliffs_delta": cliffs_delta(a, b), "n": len(d),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sensitivity_paired_stats.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
