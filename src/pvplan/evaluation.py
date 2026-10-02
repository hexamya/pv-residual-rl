"""Paired evaluation protocol (decision D13).

Every policy is evaluated on the SAME set of held-out scenario seeds: the
environment's ScenarioSampler is reseeded per episode, and its fixed draw order
guarantees identical adoption/growth/weather realizations across policies, so
per-seed differences are paired observations (Wilcoxon signed-rank applies).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .environment import EnvConfig, PVDeploymentEnv, TranchePVDeploymentEnv

EVAL_GAMMA = 0.98


def run_episode(env: PVDeploymentEnv, policy: Callable, scenario_seed: int) -> dict:
    obs, _ = env.reset(options={"scenario_seed": scenario_seed})
    if hasattr(policy, "prepare"):
        policy.prepare(env.core, scenario_seed)
    infos: list[dict[str, Any]] = []
    done = False
    while not done:
        action = policy(env.core, obs)
        obs, reward, done, _trunc, info = env.step(action)
        infos.append(info)
    return summarize_episode(infos, scenario_seed)


def run_episode_tranche(env: TranchePVDeploymentEnv, predict: Callable,
                        scenario_seed: int) -> dict:
    """For DQN policies: `predict(obs) -> int action`. Year-end infos only."""
    obs, _ = env.reset(options={"scenario_seed": scenario_seed})
    infos = []
    done = False
    while not done:
        obs, reward, done, _trunc, info = env.step(int(predict(obs)))
        if info:
            infos.append(info)
    return summarize_episode(infos, scenario_seed)


def summarize_episode(infos: list[dict], scenario_seed: int) -> dict:
    r = np.array([i["reward"] for i in infos])
    disc = float((r * EVAL_GAMMA ** np.arange(len(r))).sum())
    dpeak_mw = np.array([i["dpeak_mw"] for i in infos])
    realized = float(infos[-1]["installed_total_mw"])
    return {
        "scenario_seed": scenario_seed,
        "return_sum": float(r.sum()),
        "return_disc": disc,
        "dpeak_frac_mean": float(np.mean([i["dpeak_frac"] for i in infos])),
        "dpeak_frac_final": float(infos[-1]["dpeak_frac"]),
        "dpeak_mw_final": float(dpeak_mw[-1]),
        "overload_delta_sum": float(np.sum([i["overload_delta"] for i in infos])),
        "volt_delta_sum": float(np.sum([i["volt_delta"] for i in infos])),
        "volt_gross_sum": float(np.sum([i.get("volt_gross_frac", np.nan) for i in infos])),
        **{f"{k}_delta_sum": float(np.sum([i[f"{k}_net"] - i[f"{k}_gross"] for i in infos]))
           for k in ("uv_deficit", "uv094", "uv096") if f"{k}_net" in infos[0]},
        "uv_deficit_gross_sum": float(np.sum([i.get("uv_deficit_gross", np.nan) for i in infos])),
        "tap_ops_per_day": float(np.mean([i.get("tap_ops_per_day", np.nan) for i in infos])),
        "overload_frac_sum": float(np.sum([i["overload_frac"] for i in infos])),
        "rpf_max_mw": float(np.max([i["reverse_flow_mw"] for i in infos])),
        "offered_total_mw": float(np.sum([i["offered_mw"] for i in infos])),
        "installed_total_mw": realized,
        "budget_efficiency": float(dpeak_mw[-1] / max(realized, 1e-9)),
    }


def evaluate_policy(
    name: str,
    policy: Callable,
    pv_days: np.ndarray,
    cfg: EnvConfig,
    scenario_seeds: list[int],
    tranche: bool = False,
    n_tranches: int = 8,
) -> pd.DataFrame:
    if tranche:
        env = TranchePVDeploymentEnv(pv_days, cfg, n_tranches=n_tranches)
        rows = [run_episode_tranche(env, policy, s) for s in scenario_seeds]
    else:
        env = PVDeploymentEnv(pv_days, cfg)
        rows = [run_episode(env, policy, s) for s in scenario_seeds]
    df = pd.DataFrame(rows)
    df.insert(0, "policy", name)
    return df
