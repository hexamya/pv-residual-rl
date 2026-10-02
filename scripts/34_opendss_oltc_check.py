"""Exact (OpenDSS) re-evaluation of the regulated weak feeder.

As in 27_opendss_exact_return.py, test episodes are replayed with identical
draws and every bus-hour, with and without PV, is solved in OpenDSS. The
substation OLTC follows the same hourly rule as Network33.oltc_v0, but its
line-drop compensation acts on the exact head power (including losses): the
source voltage is updated from the solved head flow until the tap no longer
changes.

Usage: python scripts/34_opendss_oltc_check.py [--settings 1.02] [--scenarios 100] [--workers 4]
Outputs: results/opendss_oltc.csv, opendss_oltc_stats.csv
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
POLICIES = (["max-exp-cap", "max-exp-cap-V retuned"]
            + [f"residual-rt-s{s}" for s in range(3)] + [f"ppo-rt-s{s}" for s in range(3)])


def solve_hour(net, p: np.ndarray, q: np.ndarray, v0: float) -> tuple[np.ndarray, float, float]:
    """Solve one snapshot with the OLTC; returns (V_pu at buses 2..33, head kW, tap V0)."""
    import opendssdirect as dss
    for i in range(32):
        dss.Loads.Name(f"load{i + 2}")
        dss.Loads.kW(float(p[i]))
        dss.Loads.kvar(float(q[i]))
    for _ in range(6):
        dss.Text.Command(f"Vsource.source.pu={v0}")
        dss.Solution.Solve()
        assert dss.Solution.Converged()
        s = dss.Circuit.TotalPower()
        new = float(net.oltc_v0(-s[0] / 1000.0, -s[1] / 1000.0))
        if abs(new - v0) < 1e-9:
            break
        v0 = new
    v = []
    for b in range(2, 34):
        dss.Circuit.SetActiveBus(f"bus{b}")
        v.append(np.mean(dss.Bus.puVmagAngle()[::2]))
    return np.array(v), float(-s[0]), v0


def solve_case(net, p: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """Mean daily head peak (kW), share of bus-hours outside the band, mean tap V0."""
    from pvplan.network import V_MAX_PU, V_MIN_PU
    D = p.shape[1]
    heads, taps, bad, v0 = np.zeros((D, 24)), [], 0, 1.0
    for d in range(D):
        for h in range(24):
            v, heads[d, h], v0 = solve_hour(net, p[:, d, h], q[:, d, h], v0)
            taps.append(v0)
            bad += int(((v < V_MIN_PU) | (v > V_MAX_PU)).sum())
    return heads.max(axis=1).mean(), bad / (32 * D * 24), float(np.mean(taps))


def _scenario(job: tuple) -> list[dict]:
    import torch
    from pvplan.network import Network33
    from pvplan.opendss_validate import build_circuit
    from pvplan.profiles import bus_shapes
    torch.set_num_threads(1)
    ex = importlib.import_module("27_opendss_exact_return")
    reg = importlib.import_module("33_voltage_regulation")
    setting, scenario = job
    cfg = reg.cfg(setting)
    net = Network33(impedance_scale=1.0, oltc_vmax=cfg.oltc_vmax)
    build_circuit(net)
    shapes = bus_shapes()
    rows, gross = [], {}
    for label in POLICIES:
        years, infos = ex.record_episode(cfg, reg._policy(setting, label), scenario)
        peak_exact = v_exact = dkw = 0.0
        for t, (inst, scale, noise, days) in enumerate(years):
            p_g, q = ex.loads(net, shapes, scale, noise)
            if t not in gross:
                gross[t] = solve_case(net, p_g, q)
            pk_g, vg, _ = gross[t]
            pk_n, vn, _ = solve_case(net, p_g - inst[:, None, None] * 1000.0 * days[None, :, :], q)
            peak_exact += (pk_g - pk_n) / pk_g
            v_exact += -0.5 * (vn - vg)
            dkw += (pk_g - pk_n) / len(years)
        tr = -0.5 * sum(i["overload_delta"] for i in infos)
        rows.append({
            "setting": setting, "scenario_seed": scenario, "policy": label,
            "return_lin": sum(i["reward"] for i in infos),
            "v_term_lin": -0.5 * sum(i["volt_delta"] for i in infos),
            "uv_gross_lin": float(np.mean([i["volt_gross_frac"] for i in infos])),
            "uv_gross_exact": float(np.mean([g[1] for g in gross.values()])),
            "tap_mean_gross": float(np.mean([g[2] for g in gross.values()])),
            "v_term_exact": v_exact, "peak_term_exact": peak_exact,
            "return_exact": peak_exact + v_exact + tr, "peak_kw_exact": dkw,
        })
    return rows


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci

    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", default=["1.02"])
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    jobs = [(s, 10_000 + k) for s in args.settings for k in range(args.scenarios)]
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        df = pd.DataFrame([r for rows in exe.map(_scenario, jobs) for r in rows])
    df.to_csv(RESULTS / "opendss_oltc.csv", index=False)

    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    comp = df.groupby(["setting", "family", "scenario_seed"]).mean(numeric_only=True)
    ref = "max-exp-cap-V retuned"
    rows = []
    for s in args.settings:
        c = comp.loc[s]
        for a, b in [("residual-rt", ref), ("ppo-rt", ref), (ref, "max-exp-cap"),
                     ("residual-rt", "ppo-rt")]:
            for col in ["return_lin", "return_exact"]:
                x, y = c.loc[a][col], c.loc[b][col].loc[c.loc[a].index]
                d = (x - y).values
                lo, hi = bootstrap_ci(d)
                rows.append({"setting": s, "policy": a, "reference": b, "metric": col,
                             "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
                             "wilcoxon_p": 1.0 if np.allclose(d, 0)
                             else float(sps.wilcoxon(x.values, y.values).pvalue),
                             "wins": int((d > 0).sum())})
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "opendss_oltc_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.groupby(["setting", "family"])[["return_lin", "return_exact", "uv_gross_lin",
                                              "uv_gross_exact", "tap_mean_gross"]].mean().round(4).to_string())
    print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
