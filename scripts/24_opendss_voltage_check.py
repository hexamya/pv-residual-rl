"""Re-evaluate the weak-feeder voltage term with OpenDSS instead of LinDistFlow.

LinDistFlow is optimistic on the weak feeder (scripts/23_validate_feeders.py),
so the policies' undervoltage counts are recomputed with the exact power flow:
each episode is replayed with the same random draws, the per-bus installed PV,
load scale, load noise and sampled days are recorded per year, and every
bus-hour is solved in OpenDSS with and without PV. The PV-attributable voltage
term -0.5 * sum_t (V_net - V_gross) is then compared between policies.

Usage: python scripts/24_opendss_voltage_check.py [--scenarios 20] [--workers 7]
Output: results/opendss_voltage_check.csv, opendss_voltage_check_stats.csv
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
POLICIES = ["max-exp-cap-V", "residual-s0", "ppo-s0", "rolling-lp-V", "max-exp-cap"]


def make(label: str):
    import json
    weak = importlib.import_module("17_weak_feeder")
    hyb = importlib.import_module("18_hybrid")
    tuned = json.loads((RESULTS / "weak_tuned.json").read_text())
    if label.startswith("residual-s"):
        return hyb.HybridPolicy("residual", int(label[-1]))
    if label == "max-exp-cap-V":
        return weak.make_policy("max-exp-cap", tuned["kappa"])
    if label == "max-exp-cap":
        return weak.make_policy("max-exp-cap", 0.0)
    if label == "rolling-lp-V":
        return weak.make_policy("rolling-lp-v", tuned["lam_v"])
    return weak.make_policy(label)


def record_episode(policy, scenario: int) -> tuple[list, list]:
    """Run one episode; return per-year (installed, scale, noise, days) and infos."""
    from pvplan import pvgis
    from pvplan.environment import EnvConfig, PVDeploymentEnv
    env = PVDeploymentEnv(pvgis.load_summer_days(), EnvConfig(impedance_scale=1.0))
    obs, _ = env.reset(options={"scenario_seed": scenario})
    if hasattr(policy, "prepare"):
        policy.prepare(env.core, scenario)
    s = env.core.sampler
    rec: dict = {}
    noise_fn, days_fn = s.load_noise, s.rep_days
    s.load_noise = lambda n: rec.__setitem__("noise", noise_fn(n)) or rec["noise"]
    s.rep_days = lambda: rec.__setitem__("days", days_fn()) or rec["days"]
    years, infos, done = [], [], False
    while not done:
        obs, _, done, _, info = env.step(policy(env.core, obs))
        years.append((env.core.installed_mw.copy(), env.core.load_scale,
                      rec["noise"], rec["days"]))
        infos.append(info)
    return years, infos


def dss_violation_share(net, p: np.ndarray, q: np.ndarray) -> float:
    from pvplan.network import V_MAX_PU, V_MIN_PU
    from pvplan.opendss_validate import solve_state
    bad = 0
    for k in range(p.shape[1]):
        v, _ = solve_state(p[:, k], q[:, k])
        bad += int(((v < V_MIN_PU) | (v > V_MAX_PU)).sum())
    return bad / p.size


def _scenario(scenario: int) -> list[dict]:
    import torch
    from pvplan.network import Network33
    from pvplan.opendss_validate import build_circuit
    torch.set_num_threads(1)
    net = Network33(impedance_scale=1.0)
    build_circuit(net)
    rows, gross_cache = [], {}
    for label in POLICIES:
        years, infos = record_episode(make(label), scenario)
        v_net, v_gross = [], []
        for t, (inst, scale, noise, days) in enumerate(years):
            p_g = net.p_load_kw[:, None, None] * _shapes()[:, None, :] * scale * noise
            q = net.q_load_kvar[:, None, None] * _shapes()[:, None, :] * scale * noise
            p_n = p_g - inst[:, None, None] * 1000.0 * days[None, :, :]
            P_g, P_n, Q = (x.reshape(32, -1) for x in (p_g, p_n, q))
            if t not in gross_cache:
                gross_cache[t] = dss_violation_share(net, P_g, Q)
            v_gross.append(gross_cache[t])
            v_net.append(dss_violation_share(net, P_n, Q))
        rows.append({
            "scenario_seed": scenario, "policy": label,
            "v_term_lin": -0.5 * sum(i["volt_delta"] for i in infos),
            "v_term_dss": -0.5 * float(np.sum(np.array(v_net) - np.array(v_gross))),
            "viol_net_share_lin": float(np.mean([i["volt_violation_frac"] for i in infos])),
            "viol_net_share_dss": float(np.mean(v_net)),
            "return_lin": float(sum(i["reward"] for i in infos)),
        })
    return rows


_SHAPES = None


def _shapes():
    global _SHAPES
    if _SHAPES is None:
        from pvplan.profiles import bus_shapes
        _SHAPES = bus_shapes()
    return _SHAPES


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=20)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    seeds = [10_000 + k for k in range(args.scenarios)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        df = pd.DataFrame([r for rows in ex.map(_scenario, seeds) for r in rows])
    df.to_csv(RESULTS / "opendss_voltage_check.csv", index=False)

    out = []
    for ref in ["max-exp-cap-V"]:
        for pol in [p for p in POLICIES if p != ref]:
            for col in ["v_term_lin", "v_term_dss"]:
                x = df[df.policy == pol].set_index("scenario_seed")[col].sort_index()
                y = df[df.policy == ref].set_index("scenario_seed")[col].loc[x.index]
                d = (x - y).values
                lo, hi = bootstrap_ci(d)
                out.append({"policy": pol, "reference": ref, "metric": col,
                            "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                            "wilcoxon_p": float(sps.wilcoxon(x, y).pvalue)
                            if not np.allclose(d, 0) else 1.0,
                            "wins": int((d > 0).sum()), "n": len(d),
                            "rrb": rank_biserial_paired(x.values, y.values)})
    st = pd.DataFrame(out)
    st.to_csv(RESULTS / "opendss_voltage_check_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("policy")[["v_term_lin", "v_term_dss", "viol_net_share_lin",
                                "viol_net_share_dss"]].mean().round(4).to_string())
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
