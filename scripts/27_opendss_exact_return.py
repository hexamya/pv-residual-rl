"""Exact (OpenDSS) peak and voltage terms, including line losses.

Proposition 1 (location invariance of the peak term) holds for the lossless
LinDistFlow model. Here every test episode of the main policies is replayed
with identical draws, and every bus-hour, with and without PV, is solved in
OpenDSS. The feeder-head power (loads plus losses) gives the exact peak term,
and the bus voltages give the exact voltage term. The transformer term does
not depend on the power flow and is taken from the episode.

Usage: python scripts/27_opendss_exact_return.py [--scenarios 100] [--workers 7]
Outputs: results/opendss_exact_return.csv, opendss_exact_stats.csv
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "results"
POLICIES = {
    "stiff": ["max-exp-cap", "accept-greedy", "rolling-lp", "peak-first"]
             + [f"ppo-s{s}" for s in range(5)],
    "weak": ["max-exp-cap-V", "max-exp-cap", "rolling-lp-V"]
            + [f"ppo-s{s}" for s in range(5)] + [f"residual-s{s}" for s in range(5)],
}
IMPEDANCE = {"stiff": 0.5, "weak": 1.0}


def make(feeder: str, label: str):
    import json
    from pvplan.baselines import (AcceptGreedyPolicy, MaxExpectedCapacityPolicy,
                                  PeakFirstPolicy, RollingLPPolicy, RollingLPVPolicy)
    weak = importlib.import_module("17_weak_feeder")
    hyb = importlib.import_module("18_hybrid")
    tuned = json.loads((RESULTS / "weak_tuned.json").read_text())
    if label.startswith("ppo-s"):
        tag = "nominal" if feeder == "stiff" else "weak"
        best = RESULTS / "logs" / tag / f"ppo_seed{label[-1]}" / "best_model.zip"
        return weak.SB3Policy(best)
    if label.startswith("residual-s"):
        return hyb.HybridPolicy("residual", int(label[-1]))
    return {"max-exp-cap": lambda: MaxExpectedCapacityPolicy(0.0),
            "max-exp-cap-V": lambda: MaxExpectedCapacityPolicy(tuned["kappa"]),
            "accept-greedy": AcceptGreedyPolicy, "peak-first": PeakFirstPolicy,
            "rolling-lp": RollingLPPolicy,
            "rolling-lp-V": lambda: RollingLPVPolicy(tuned["lam_v"])}[label]()


def record_episode(cfg, policy, scenario: int):
    from pvplan import pvgis
    from pvplan.environment import PVDeploymentEnv
    env = PVDeploymentEnv(pvgis.load_summer_days(), cfg)
    obs, _ = env.reset(options={"scenario_seed": scenario})
    s = env.core.sampler
    rec: dict = {}
    noise_fn, days_fn = s.load_noise, s.rep_days
    s.load_noise = lambda n: rec.__setitem__("noise", noise_fn(n)) or rec["noise"]
    s.rep_days = lambda: rec.__setitem__("days", days_fn()) or rec["days"]
    years, infos, done = [], [], False
    while not done:
        obs, _, done, _, info = env.step(policy(env.core, obs))
        years.append((env.core.installed_mw.copy(), env.core.load_scale, rec["noise"], rec["days"]))
        infos.append(info)
    return years, infos


def solve_case(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Mean daily feeder-head peak (kW) and share of bus-hours outside the band."""
    from pvplan.network import V_MAX_PU, V_MIN_PU
    from pvplan.opendss_validate import solve_state
    D = p.shape[1]
    heads = np.zeros((D, 24))
    bad = 0
    for d in range(D):
        for h in range(24):
            v, hk = solve_state(p[:, d, h], q[:, d, h])
            heads[d, h] = hk
            bad += int(((v < V_MIN_PU) | (v > V_MAX_PU)).sum())
    return heads.max(axis=1).mean(), bad / (32 * D * 24)


def loads(net, shapes, scale, noise):
    p_g = net.p_load_kw[:, None, None] * shapes[:, None, :] * scale * noise
    q = net.q_load_kvar[:, None, None] * shapes[:, None, :] * scale * noise
    return p_g, q


def solve_year(net, shapes, inst, scale, noise, days) -> tuple[float, float, float, float]:
    """Mean daily head peak (gross, net) in kW and violation shares (gross, net)."""
    p_g, q = loads(net, shapes, scale, noise)
    p_n = p_g - inst[:, None, None] * 1000.0 * days[None, :, :]
    (pk_g, v_g), (pk_n, v_n) = solve_case(p_g, q), solve_case(p_n, q)
    return pk_g, pk_n, v_g, v_n


def _scenario(job: tuple) -> list[dict]:
    import torch
    from pvplan.environment import EnvConfig
    from pvplan.network import Network33
    from pvplan.opendss_validate import build_circuit
    from pvplan.profiles import bus_shapes
    torch.set_num_threads(1)
    feeder, scenario = job
    cfg = EnvConfig(impedance_scale=IMPEDANCE[feeder])
    net = Network33(impedance_scale=IMPEDANCE[feeder])
    build_circuit(net)
    shapes = bus_shapes()
    rows, gross = [], {}
    for label in POLICIES[feeder]:
        years, infos = record_episode(cfg, make(feeder, label), scenario)
        peak_exact = v_exact = dkw = 0.0
        for t, (inst, scale, noise, days) in enumerate(years):
            p_g, q = loads(net, shapes, scale, noise)     # identical draws for every policy
            if t not in gross:
                gross[t] = solve_case(p_g, q)
            pk_g, vg = gross[t]
            pk_n, vn = solve_case(p_g - inst[:, None, None] * 1000.0 * days[None, :, :], q)
            peak_exact += (pk_g - pk_n) / pk_g
            v_exact += -0.5 * (vn - vg)
            dkw += (pk_g - pk_n) / len(years)
        tr = -0.5 * sum(i["overload_delta"] for i in infos)
        rows.append({
            "feeder": feeder, "scenario_seed": scenario, "policy": label,
            "peak_term_lin": sum(i["dpeak_frac"] for i in infos),
            "v_term_lin": -0.5 * sum(i["volt_delta"] for i in infos),
            "return_lin": sum(i["reward"] for i in infos),
            "peak_term_exact": peak_exact, "v_term_exact": v_exact,
            "return_exact": peak_exact + v_exact + tr,
            "peak_kw_exact": dkw, "installed_mw": float(years[-1][0].sum()),
        })
    return rows


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--feeders", nargs="*", default=["stiff", "weak"])
    args = ap.parse_args()
    jobs = [(f, 10_000 + k) for f in args.feeders for k in range(args.scenarios)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        df = pd.DataFrame([r for rows in ex.map(_scenario, jobs) for r in rows])
    path = RESULTS / "opendss_exact_return.csv"
    if path.exists():                                     # keep the other feeder's rows
        old = pd.read_csv(path)
        df = pd.concat([old[~old.feeder.isin(args.feeders)], df], ignore_index=True)
    df.to_csv(path, index=False)

    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    comp = df.groupby(["feeder", "family", "scenario_seed"]).mean(numeric_only=True).reset_index()
    pairs = {"stiff": [("max-exp-cap", "ppo"), ("max-exp-cap", "rolling-lp"),
                       ("accept-greedy", "ppo"), ("ppo", "rolling-lp")],
             "weak": [("residual", "max-exp-cap-V"), ("ppo", "max-exp-cap-V"),
                      ("rolling-lp-V", "max-exp-cap-V"), ("max-exp-cap-V", "max-exp-cap")]}
    rows = []
    for feeder, plist in pairs.items():
        c = comp[comp.feeder == feeder]
        if c.empty:
            continue
        for a, b in plist:
            for col in ["return_lin", "return_exact", "peak_term_lin", "peak_term_exact",
                        "v_term_exact", "peak_kw_exact"]:
                x = c[c.family == a].set_index("scenario_seed")[col].sort_index()
                y = c[c.family == b].set_index("scenario_seed")[col].loc[x.index]
                d = (x - y).values
                lo, hi = bootstrap_ci(d)
                rows.append({"feeder": feeder, "policy": a, "reference": b, "metric": col,
                             "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": 1.0 if np.allclose(d, 0)
                             else float(sps.wilcoxon(x.values, y.values).pvalue),
                             "wins": int((d > 0).sum()), "rrb": rank_biserial_paired(x.values, y.values)})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "opendss_exact_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(comp.groupby(["feeder", "family"])[["return_lin", "return_exact", "peak_term_lin",
                                               "peak_term_exact", "peak_kw_exact"]].mean().round(4).to_string())
    print(st[st.metric.isin(["return_lin", "return_exact", "peak_kw_exact"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
