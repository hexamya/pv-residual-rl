"""Expected-capacity allocation layer used by the hybrid and residual policies.

Given acceptance xi_i ~ Beta(a_i, b_i), remaining rooftop headroom H_i and
priority weights w_i, the annual budget B is split into offers o_i solving

    max_o  sum_i w_i E[min(o_i xi_i, H_i)]   s.t.  sum_i o_i = B,  o >= 0.

The marginal value w_i mean_i I_{H_i/o_i}(a_i + 1, b_i) is flat up to o_i = H_i
and decreasing beyond, so the KKT solution follows from a bisection on the
budget multiplier nu: offers are H_i / I^{-1}(nu / (w_i mean_i)) for buses with
w_i mean_i > nu, and the budget left at the threshold goes to the buses whose
marginal equals nu, in proportion to their headroom.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import betaincinv

from .network import V_MIN_PU


def acceptance_params(core: Any) -> tuple[np.ndarray, np.ndarray]:
    """Beta parameters of this year's acceptance, as the sampler will draw it."""
    s = core.sampler
    n = len(core.installed_mw)
    a, b = s._acc_a[:n], s._acc_b[:n]
    if s.cfg.adoption_model == "bass":
        conc = a + b
        mean = np.clip(a / conc * s.cfg.bass_p_scale
                       + s.cfg.bass_q * core.installed_mw / core.pot_mw, 0.02, 0.98)
        a, b = mean * conc, (1 - mean) * conc
    return a, b


def voltage_relief_weights(core: Any) -> np.ndarray:
    """Undervoltage relief per MW at each bus on next year's expected nominal day.

    s_i = sum over bus-hours (n, h) below V_min of Rc[n, i] * pv(h), evaluated
    with the installed PV; scaled to max 1 (zeros if no bus-hour is below V_min).
    """
    scale = core.load_scale * (1.0 + core.cfg.uncertainty.growth_mean)
    p = (core.p_nom_kw[:, None] * core.shapes * scale
         - core.installed_mw[:, None] * 1000.0 * core.pv_mean_day[None, :])
    q = core.q_nom_kvar[:, None] * core.shapes * scale
    below = core.net.voltages_pu(p, q) < V_MIN_PU                  # (n, 24)
    s = core.net.r_common_pu.T @ (below * core.pv_mean_day[None, :]).sum(axis=1)
    return s / s.max() if s.max() > 0 else s


def expected_capacity_allocation(a: np.ndarray, b: np.ndarray, head: np.ndarray,
                                 budget: float, weight: np.ndarray | None = None,
                                 iters: int = 60) -> np.ndarray:
    n = len(a)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
    g0 = w * a / (a + b) * (head > 0)                     # marginal value at o = 0
    if budget <= 0 or g0.max() <= 0:
        return np.full(n, max(budget, 0.0) / n)

    def offers(nu: float) -> np.ndarray:
        o = np.zeros(n)
        act = g0 > nu
        c = betaincinv(a[act] + 1, b[act], nu / g0[act])
        o[act] = head[act] / np.maximum(c, 1e-12)
        return o

    lo, hi = 0.0, float(g0.max())         # offers: unbounded at lo, zero at hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if offers(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
    o = offers(hi)
    rem = budget - o.sum()
    margin = (g0 > lo) & (g0 <= hi)       # flat-marginal buses at the threshold
    if rem > 0 and margin.any():
        o[margin] += rem * head[margin] / head[margin].sum()
    elif rem > 0:
        o[o > 0] *= budget / o.sum()
    return o


def expected_installed(a: np.ndarray, b: np.ndarray, head: np.ndarray,
                       offered: np.ndarray, n_mc: int = 20_000,
                       seed: int = 0) -> np.ndarray:
    """Monte Carlo E[min(o_i xi_i, H_i)] per bus (for tests)."""
    rng = np.random.default_rng(seed)
    xi = rng.beta(a, b, size=(n_mc, len(a)))
    return np.minimum(offered * xi, head).mean(axis=0)
