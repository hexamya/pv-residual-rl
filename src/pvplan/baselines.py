"""Baseline allocation policies and the LP planner (decision D12).

All policies return an *action* for PVDeploymentEnv (raw logits, length 33:
[reserve, bus1..bus32]); softmax(log(shares)) == shares, so a policy that wants
allocation shares s just returns log(s + tiny).

Baselines (proposal 8-5-1, with corrections):
  B1 random          — Dirichlet-random budget split (lower bound);
  B2 peak-first      — proportional to current local net peak (common practice);
  B3 cf-static       — proportional to the t=0 coincidence factor (proposal);
  B4 cf-adaptive     — proportional to the *current* dynamic CF;
  B5 rolling-lp      — deterministic multi-period LP (expected acceptance/growth)
                       re-solved each year, first-year decision applied (MPC);
  ORACLE             — the same LP with *realized* adoption & growth sequences
                       (perfect information) -> optimality-gap upper reference.

Structured rules (the peak term depends only on total MW):
  accept-greedy      — offers in decreasing order of mean acceptance;
  max-exp-cap        — myopic exact allocation maximising expected installed
                       capacity, optionally weighted by undervoltage relief (kappa);
  rolling-lp-v       — rolling LP with a LinDistFlow undervoltage term.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from .allocation import (acceptance_params, expected_capacity_allocation,
                         voltage_relief_weights)
from .environment import N_BUS, PlanningCore
from .network import V_MIN_PU
from .uncertainty import ScenarioSampler

TINY = 1e-9


def shares_to_action(shares: np.ndarray, reserve: float = 0.0) -> np.ndarray:
    s = np.concatenate([[reserve], shares])
    s = np.maximum(s, 0)
    s = s / max(s.sum(), TINY)
    return np.log(s + TINY).astype(np.float32)


def _offered_to_action(offered: np.ndarray, budget: float) -> np.ndarray:
    """Convert an offered-MW vector to env action; leftover budget -> reserve."""
    tot = offered.sum()
    frac = min(1.0, tot / max(budget, TINY))
    shares = offered / max(tot, TINY) * frac
    return shares_to_action(shares, reserve=max(0.0, 1.0 - frac))


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        return shares_to_action(self.rng.dirichlet(np.ones(N_BUS)))


class PeakFirstPolicy:
    """Budget proportional to each neighborhood's current local net peak."""

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        peaks = core._nominal_net_kw().max(axis=1)
        return shares_to_action(np.maximum(peaks, 0))


class CFPolicy:
    """Budget proportional to CF (static: frozen at t=0; adaptive: current)."""

    def __init__(self, static: bool = True):
        self.static = static
        self._cf0: np.ndarray | None = None

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        if core.t == 0 or self._cf0 is None:
            self._cf0 = np.maximum(core.dynamic_cf(), 0)
        cf = self._cf0 if self.static else np.maximum(core.dynamic_cf(), 0)
        if cf.sum() < 1e-6:
            cf = np.ones(N_BUS)
        return shares_to_action(cf)


# ---------------------------------------------------------------------------
# LP planner: multi-period peak-epigraph minimization
#   min sum_t gamma^t z_t
#   s.t. z_t >= sys_gross(t,h) - pv_existing(h) - sum_i pvshape(h)*a[i,tt]*x[i,tt<=t]
#        sum_i x[i,t] <= B ; cumulative expected installs <= headroom ; x >= 0
# ---------------------------------------------------------------------------
def solve_plan(core: PlanningCore, accept: np.ndarray, growth: np.ndarray,
               gamma: float = 0.98) -> np.ndarray:
    """LP over `len(growth)` periods from the core's current state.

    accept: (T, N) acceptance rates assumed by the planner.
    Returns the (T, N) offered-MW plan.
    """
    cfg = core.cfg
    T = len(growth)
    pvshape = core.pv_mean_day * 1000.0                     # kW per MW installed
    p0 = core.p_nom_kw * core.load_scale
    scale = np.cumprod(1.0 + growth)
    gross_sys = np.outer(scale, (p0[:, None] * core.shapes).sum(axis=0))  # (T,24)
    existing_kw = core.installed_mw.sum() * pvshape         # (24,)

    n_var = N_BUS * T + T
    c = np.zeros(n_var)
    c[N_BUS * T:] = gamma ** np.arange(T)

    A_ub, b_ub = [], []
    for t in range(T):
        for h in range(24):
            row = np.zeros(n_var)
            for tt in range(t + 1):
                row[tt * N_BUS:(tt + 1) * N_BUS] = -pvshape[h] * accept[tt]
            row[N_BUS * T + t] = -1.0
            A_ub.append(row)
            b_ub.append(-(gross_sys[t, h] - existing_kw[h]))
        row = np.zeros(n_var)
        row[t * N_BUS:(t + 1) * N_BUS] = 1.0
        A_ub.append(row)
        b_ub.append(cfg.annual_budget_mw)
    headroom = np.maximum(core.pot_mw - core.installed_mw, 0.0)
    for i in range(N_BUS):
        row = np.zeros(n_var)
        for tt in range(T):
            row[tt * N_BUS + i] = accept[tt, i]
        A_ub.append(row)
        b_ub.append(headroom[i])

    bounds = [(0, None)] * (N_BUS * T) + [(None, None)] * T
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds,
                  method="highs")
    if not res.success:
        return np.full((T, N_BUS), cfg.annual_budget_mw / N_BUS)
    return res.x[:N_BUS * T].reshape(T, N_BUS)


class RollingLPPolicy:
    """MPC baseline: deterministic LP with expected parameters, re-solved yearly."""

    def __init__(self, gamma: float = 0.98):
        self.gamma = gamma

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        T_rem = core.cfg.horizon_years - core.t
        a_bar = core.sampler._acc_a / (core.sampler._acc_a + core.sampler._acc_b)
        accept = np.tile(a_bar, (T_rem, 1))
        growth = np.full(T_rem, core.cfg.uncertainty.growth_mean)
        plan = solve_plan(core, accept, growth, self.gamma)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        return _offered_to_action(plan[0], budget)


class OraclePolicy:
    """Perfect-information LP: sees the episode's realized adoption & growth.

    Realized sequences are pre-extracted by replaying the episode's
    ScenarioSampler seed with the environment's fixed draw order (acceptance,
    growth, load-noise, rep-days per year). Day-level weather enters at its
    expectation, so this is an upper reference w.r.t. adoption/growth
    uncertainty (documented in D12).
    """

    def __init__(self, gamma: float = 0.98):
        self.gamma = gamma
        self._plan: np.ndarray | None = None    # (T, N) offered plan

    def prepare(self, core: PlanningCore, scenario_seed: int) -> None:
        cfg = core.cfg
        probe = ScenarioSampler(core.pv_days_all, cfg.uncertainty, scenario_seed)
        T = cfg.horizon_years
        accept = np.zeros((T, N_BUS))
        growth = np.zeros(T)
        for t in range(T):
            accept[t] = probe.acceptance(N_BUS)
            growth[t] = probe.demand_growth()
            probe.load_noise(N_BUS)   # keep draw order aligned with apply_year
            probe.rep_days()
        self._plan = solve_plan(core, accept, growth, self.gamma)

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        assert self._plan is not None, "call prepare(core, scenario_seed) first"
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        return _offered_to_action(self._plan[core.t], budget)


# ---------------------------------------------------------------------------
# Acceptance-aware rules. Both spend the whole budget
# and use only the acceptance distributions and the remaining rooftop headroom.
# ---------------------------------------------------------------------------
class AcceptGreedyPolicy:
    """Offer the budget in decreasing order of mean acceptance, each
    neighbourhood up to its expected headroom (headroom / mean acceptance)."""

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        a, b = acceptance_params(core)
        mean = a / (a + b)
        head = np.maximum(core.pot_mw - core.installed_mw, 0.0)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        offered = np.zeros(N_BUS)
        rem = budget
        for i in np.lexsort((-head, -mean)):      # mean desc, then headroom desc
            if rem <= 0:
                break
            o = min(rem, head[i] / mean[i])
            offered[i], rem = o, rem - o
        if rem > 1e-12 and head.sum() > 0:        # all headroom covered
            offered += rem * head / head.sum()
        return _offered_to_action(offered, budget)


class MaxExpectedCapacityPolicy:
    """Myopic rule: split the budget to maximise the capacity expected to be
    installed this year, sum_i w_i E[min(o_i xi_i, H_i)], solved exactly
    (allocation.expected_capacity_allocation). With kappa > 0 the weights are
    w_i = 1 + kappa * s_i, s_i = voltage_relief_weights at the year's start."""

    def __init__(self, kappa: float = 0.0):
        self.kappa = kappa

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        a, b = acceptance_params(core)
        head = np.maximum(core.pot_mw - core.installed_mw, 0.0)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        weight = 1.0 + self.kappa * voltage_relief_weights(core) if self.kappa else None
        offered = expected_capacity_allocation(a, b, head, budget, weight)
        return _offered_to_action(offered, budget)


class HedgedCapacityPolicy:
    """Max-expected-capacity rule hedged against acceptance-model errors.

    Offers = (1 - eta) * exact allocation + eta * spread, where the spread
    allocates the budget in proportion to w_i * H_i (weight times headroom).
    The exact allocation puts the budget on the buses that look best and is
    therefore fragile when the believed ranking of acceptance is wrong."""

    def __init__(self, kappa: float = 0.0, eta: float = 0.2):
        self.kappa, self.eta = kappa, eta

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        a, b = acceptance_params(core)
        head = np.maximum(core.pot_mw - core.installed_mw, 0.0)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        weight = (1.0 + self.kappa * voltage_relief_weights(core) if self.kappa
                  else np.ones(N_BUS))
        exact = expected_capacity_allocation(a, b, head, budget, weight)
        wh = weight * head
        spread = budget * wh / wh.sum() if wh.sum() > 0 else exact
        return _offered_to_action((1 - self.eta) * exact + self.eta * spread, budget)


class AdaptiveCapacityPolicy:
    """Max-expected-capacity rule that learns class acceptance from observed uptake.

    The believed class means (acceptance_params) act as a prior worth `n0` MW of
    offers. Each year the policy compares last year's offers with the capacity
    actually installed. For buses whose uptake was not capped by rooftop
    headroom, the accepted and offered MW are added to their class totals, and
    the class mean becomes (n0 * prior + accepted) / (n0 + offered). Offers then
    follow the exact allocation with the updated means (concentration kept)."""

    def __init__(self, kappa: float = 0.0, n0: float = 0.1):
        from .profiles import class_labels
        self.kappa, self.n0 = kappa, n0
        self.cls = np.array(class_labels())
        self._reset()

    def _reset(self) -> None:
        self.acc = {c: 0.0 for c in set(self.cls)}
        self.off = {c: 0.0 for c in set(self.cls)}
        self.last = None                         # (offers, installed, headroom)

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        if core.t == 0:
            self._reset()
        head = np.maximum(core.pot_mw - core.installed_mw, 0.0)
        if self.last is not None:
            offers, inst0, head0 = self.last
            got = core.installed_mw - inst0
            uncapped = (offers > 1e-9) & (got < head0 - 1e-9)
            for c in self.acc:
                m = uncapped & (self.cls == c)
                self.acc[c] += float(got[m].sum())
                self.off[c] += float(offers[m].sum())
        a, b = acceptance_params(core)
        conc = a + b
        mean = a / conc
        for c in self.acc:
            m = self.cls == c
            mean[m] = (self.n0 * mean[m] + self.acc[c]) / (self.n0 + self.off[c])
        mean = np.clip(mean, 0.02, 0.98)
        a, b = mean * conc, (1 - mean) * conc
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        weight = 1.0 + self.kappa * voltage_relief_weights(core) if self.kappa else None
        offered = expected_capacity_allocation(a, b, head, budget, weight)
        self.last = (offered.copy(), core.installed_mw.copy(), head)
        return _offered_to_action(offered, budget)


# ---------------------------------------------------------------------------
# LP planner with an undervoltage term (LinDistFlow is linear in PV):
#   min sum_t gamma^t [ z_t / Pg_t + lam_v * sum_{n,h} u[n,t,h] ]
#   s.t. peak epigraph as in solve_plan,
#        u[n,t,h] >= V_min - Vg[n,t,h] - pv(h) * sum_m Rc[n,m] X[m,t],  u >= 0
#   X[m,t] = installed_m + sum_{tt<=t} accept[tt,m] * x[m,tt]
# Only PV-producing hours enter the voltage block.
# ---------------------------------------------------------------------------
def solve_plan_v(core: PlanningCore, accept: np.ndarray, growth: np.ndarray,
                 lam_v: float, gamma: float = 0.98) -> np.ndarray:
    cfg = core.cfg
    T = len(growth)
    pvshape = core.pv_mean_day * 1000.0                     # kW per MW installed
    scale = core.load_scale * np.cumprod(1.0 + growth)      # (T,)
    p_bus = core.p_nom_kw[:, None] * core.shapes            # (32, 24) at scale 1
    q_bus = core.q_nom_kvar[:, None] * core.shapes
    gross_sys = np.outer(scale, p_bus.sum(axis=0))          # (T, 24)
    existing_kw = core.installed_mw.sum() * pvshape
    hrs = np.where(core.pv_mean_day > 1e-6)[0]
    H = len(hrs)
    rc = core.net.r_common_pu

    nx, nz, nu = N_BUS * T, T, N_BUS * T * H
    n_var = nx + nz + nu
    disc = gamma ** np.arange(T)
    c = np.zeros(n_var)
    c[nx:nx + nz] = disc / gross_sys.max(axis=1)
    c[nx + nz:] = lam_v * np.repeat(disc, H * N_BUS)

    blocks, rhs = [], []
    # peak epigraph: -sum_{tt<=t} pv(h) a[tt] x[tt] - z_t <= -(gross - existing)
    rows, cols, vals = [], [], []
    for t in range(T):
        for h in range(24):
            r = t * 24 + h
            for tt in range(t + 1):
                rows += [r] * N_BUS
                cols += list(tt * N_BUS + np.arange(N_BUS))
                vals += list(-pvshape[h] * accept[tt])
            rows.append(r); cols.append(nx + t); vals.append(-1.0)
            rhs.append(-(gross_sys[t, h] - existing_kw[h]))
    blocks.append(coo_matrix((vals, (rows, cols)), shape=(T * 24, n_var)))
    # annual budget
    rows = np.repeat(np.arange(T), N_BUS)
    blocks.append(coo_matrix((np.ones(nx), (rows, np.arange(nx))), shape=(T, n_var)))
    rhs += [cfg.annual_budget_mw] * T
    # expected cumulative installs within headroom
    headroom = np.maximum(core.pot_mw - core.installed_mw, 0.0)
    cols = np.arange(nx)
    blocks.append(coo_matrix((accept.ravel(), (cols % N_BUS, cols)),
                             shape=(N_BUS, n_var)))
    rhs += list(headroom)
    # undervoltage: -pv(h) sum_m Rc[n,m] sum_{tt<=t} a[tt,m] x[m,tt] - u[n,t,h]
    #               <= -(V_min - Vg[n,t,h] - pv(h) sum_m Rc[n,m] installed_m)
    v_exist = rc @ core.installed_mw
    rows, cols, vals = [], [], []
    n_rows = 0
    for t in range(T):
        vg = core.net.voltages_pu(p_bus * scale[t], q_bus * scale[t])   # (32, 24)
        coef = np.concatenate([rc * accept[tt][None, :] for tt in range(t + 1)],
                              axis=1).ravel()               # (32 * 32(t+1),)
        w = N_BUS * (t + 1)
        r_x = np.repeat(np.arange(N_BUS), w)
        c_x = np.tile(np.arange(w), N_BUS)
        for j, h in enumerate(hrs):
            pv = core.pv_mean_day[h]
            rows += [n_rows + r_x, n_rows + np.arange(N_BUS)]
            cols += [c_x, nx + nz + (t * H + j) * N_BUS + np.arange(N_BUS)]
            vals += [-pv * coef, -np.ones(N_BUS)]
            rhs += list(-(V_MIN_PU - vg[:, h] - pv * v_exist))
            n_rows += N_BUS
    blocks.append(coo_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n_rows, n_var)))

    A = vstack(blocks).tocsr()
    bounds = [(0, None)] * nx + [(None, None)] * nz + [(0, None)] * nu
    res = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if not res.success:
        return np.full((T, N_BUS), cfg.annual_budget_mw / N_BUS)
    return res.x[:nx].reshape(T, N_BUS)


class RollingLPVPolicy:
    """Rolling LP with the undervoltage term; expected acceptance and growth."""

    def __init__(self, lam_v: float, gamma: float = 0.98):
        self.lam_v, self.gamma = lam_v, gamma

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        T_rem = core.cfg.horizon_years - core.t
        a, b = acceptance_params(core)
        accept = np.tile(a / (a + b), (T_rem, 1))
        growth = np.full(T_rem, core.cfg.uncertainty.growth_mean)
        plan = solve_plan_v(core, accept, growth, self.lam_v, self.gamma)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        return _offered_to_action(plan[0], budget)


# ---------------------------------------------------------------------------
# Scenario-based lookahead: one-step rollout of the rule
# ---------------------------------------------------------------------------
def rule_offers(core: PlanningCore, kappa: float) -> np.ndarray:
    """Offers of the max-exp-cap rule with voltage weight kappa (whole budget)."""
    a, b = acceptance_params(core)
    head = np.maximum(core.pot_mw - core.installed_mw, 0.0)
    budget = core.cfg.annual_budget_mw + core.carryover_mw
    weight = 1.0 + kappa * voltage_relief_weights(core) if kappa else None
    return expected_capacity_allocation(a, b, head, budget, weight)


class RolloutCapacityPolicy:
    """Lookahead version of the voltage-weighted rule (policy rollout).

    Every year each candidate weight kappa_c defines this year's allocation; the
    remaining years follow the base rule (kappa = base_kappa). The candidate with
    the highest mean undiscounted return over n_samples futures sampled from the
    model is applied. The sampled futures (acceptance, growth, irradiance, load
    noise) are drawn with seeds unrelated to the evaluated scenario, and the
    same futures are used for every candidate (common random numbers). The
    year-by-year choice of kappa lets the rule trade installation speed against
    voltage relief over time, which the myopic rule cannot do.
    """

    def __init__(self, kappas: list[float], base_kappa: float, n_samples: int = 40):
        self.kappas, self.base_kappa, self.n_samples = list(kappas), base_kappa, n_samples
        self.base_seed = 0
        self.chosen: list[float] = []

    def prepare(self, core: PlanningCore, scenario_seed: int) -> None:
        self.base_seed = 50_000_000 + 1000 * int(scenario_seed)
        self.chosen = []

    def _simulate(self, core: PlanningCore, kappa: float, seed: int) -> float:
        import copy
        sim = copy.copy(core)
        sim.installed_mw = core.installed_mw.copy()
        sim.sampler = ScenarioSampler(core.pv_days_all, core.cfg.uncertainty, seed)
        total = sim.apply_year(rule_offers(sim, kappa), 0.0)["reward"]
        while sim.t < core.cfg.horizon_years:
            total += sim.apply_year(rule_offers(sim, self.base_kappa), 0.0)["reward"]
        return total

    def __call__(self, core: PlanningCore, obs: np.ndarray) -> np.ndarray:
        seeds = [self.base_seed + 50 * core.t + k for k in range(self.n_samples)]
        values = [np.mean([self._simulate(core, kc, s) for s in seeds]) for kc in self.kappas]
        kappa = self.kappas[int(np.argmax(values))]
        self.chosen.append(kappa)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        return _offered_to_action(rule_offers(core, kappa), budget)
