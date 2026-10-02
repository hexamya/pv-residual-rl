"""Cross-validation of the LinDistFlow surrogate against OpenDSS (decision D02).

Builds the IEEE 33-bus feeder in OpenDSS (via OpenDSSDirect.py), applies random
load/PV states spanning the range the RL environment visits (0-100% potential
penetration, 0.7x-1.6x load), and compares bus voltage magnitudes and feeder
head power against the linear surrogate. The resulting error statistics are
reported in the thesis implementation chapter.
"""

from __future__ import annotations

import numpy as np
import opendssdirect as dss

from .network import BRANCH_DATA, IMPEDANCE_SCALE, V_BASE_KV, V_MIN_PU, Network33


def build_circuit(net: Network33) -> None:
    dss.Text.Command("Clear")
    dss.Text.Command(
        f"New Circuit.ieee33 basekv={V_BASE_KV} pu=1.0 phases=3 bus1=bus1 "
        "MVAsc3=20000 MVAsc1=20000"
    )
    for k, (f, t, _r, _x, p, q) in enumerate(BRANCH_DATA, start=1):
        r, x = net.branch_r_ohm[k - 1], net.branch_x_ohm[k - 1]
        dss.Text.Command(
            f"New Line.L{k} bus1=bus{f} bus2=bus{t} phases=3 "
            f"R1={r} X1={x} R0={r} X0={x} C1=0 C0=0 length=1 units=none"
        )
        dss.Text.Command(
            f"New Load.load{t} bus1=bus{t} phases=3 conn=wye model=1 "
            f"kV={V_BASE_KV} kW={p} kvar={q} vminpu=0.8 vmaxpu=1.2"
        )
    dss.Text.Command(f"Set VoltageBases=[{V_BASE_KV}]")
    dss.Text.Command("CalcVoltageBases")
    dss.Text.Command("Set mode=snapshot")


def solve_state(p_net_kw: np.ndarray, q_net_kvar: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve one snapshot; returns (V_pu at load buses 2..33, feeder head kW)."""
    for i in range(32):
        dss.Loads.Name(f"load{i + 2}")
        dss.Loads.kW(float(p_net_kw[i]))
        dss.Loads.kvar(float(q_net_kvar[i]))
    dss.Solution.Solve()
    assert dss.Solution.Converged()
    v = []
    for b in range(2, 34):
        dss.Circuit.SetActiveBus(f"bus{b}")
        v.append(np.mean(dss.Bus.puVmagAngle()[::2]))
    head_kw = -dss.Circuit.TotalPower()[0]  # TotalPower returns negative of injection
    return np.array(v), float(head_kw)


def run_validation(n_samples: int = 200, seed: int = 0,
                   impedance_scale: float = IMPEDANCE_SCALE) -> dict:
    net = Network33(impedance_scale=impedance_scale)
    rng = np.random.default_rng(seed)
    build_circuit(net)

    v_err, head_err_pct = [], []
    v_bias, below_lin, below_dss = [], [], []
    for _ in range(n_samples):
        load_scale = rng.uniform(0.7, 1.6)
        pen = rng.uniform(0.0, 1.0, size=32)
        shape = rng.uniform(0.3, 1.0, size=32)
        pv_kw = pen * net.p_load_kw * rng.uniform(0.0, 1.0)   # up to ~1x nominal
        p = net.p_load_kw * shape * load_scale - pv_kw
        q = net.q_load_kvar * shape * load_scale

        v_dss, head_dss = solve_state(p, q)
        v_lin = net.voltages_pu(p, q)
        head_lin = net.feeder_head_kw(p)
        v_err.append(np.abs(v_dss - v_lin).max())
        v_bias.append(np.mean(v_lin - v_dss))
        below_lin.append(v_lin < V_MIN_PU)
        below_dss.append(v_dss < V_MIN_PU)
        if abs(head_dss) > 50:
            head_err_pct.append(100 * abs(head_lin - head_dss) / abs(head_dss))

    bl, bd = np.concatenate(below_lin), np.concatenate(below_dss)
    return {
        "n_samples": n_samples,
        "impedance_scale": impedance_scale,
        "v_err_max_pu": float(np.max(v_err)),
        "v_err_mean_pu": float(np.mean(v_err)),
        "v_err_p95_pu": float(np.quantile(v_err, 0.95)),
        "v_bias_mean_pu": float(np.mean(v_bias)),       # >0: LinDistFlow optimistic
        "head_err_mean_pct": float(np.mean(head_err_pct)),
        "head_err_p95_pct": float(np.quantile(head_err_pct, 0.95)),
        # bus-states below V_min: share counted by OpenDSS, and the share whose
        # below/above classification LinDistFlow gets wrong
        "below_vmin_share_dss": float(bd.mean()),
        "below_vmin_share_lin": float(bl.mean()),
        "below_vmin_misclassified": float((bl != bd).mean()),
    }
