"""Gymnasium environments for dynamic rooftop-PV subsidy deployment planning.

MDP (proposal section 1-3, with corrections D06-D09):
  - one step = one annual budget period, episode = T years;
  - state: per-neighborhood [PV penetration, transformer margin, dynamic CF,
    normalized net peak] + [budget carryover, t/T];
  - action (continuous env, PPO/SAC): softmax shares of the available budget
    over 32 neighborhoods + 1 reserve component (carryover);
  - action (tranche env, DQN): the annual budget is split into M tranches and
    the agent assigns one tranche per sub-step to a neighborhood (or reserve);
  - transition: stochastic acceptance (Beta per class), demand growth (Normal,
    truncated), bootstrap of real PVGIS summer days, hourly load noise;
  - reward (D08): normalized system peak reduction minus normalized transformer
    overload and voltage-violation penalties (optional CF-shaping for ablation).

Physics: LinDistFlow sensitivity matrices (network.Network33); every annual
step evaluates n_rep_days x 24 h of network state as dense matrix algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .allocation import (acceptance_params, expected_capacity_allocation,
                         voltage_relief_weights)
from .network import Network33, V_MAX_PU, V_MIN_PU
from .profiles import bus_shapes, rho_potential
from .uncertainty import ScenarioSampler, UncertaintyConfig

N_BUS = 32
OBS_PER_BUS = 4
CF_EPS_MW = 0.05          # probe size for the marginal (dynamic) CF estimate


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def apply_bess(pv_days: np.ndarray, share: float, shift_frac: float) -> np.ndarray:
    """Shift `share*shift_frac` of daily PV energy from hours 10-16 to 18-23 (D14)."""
    if share <= 0.0:
        return pv_days
    out = pv_days.copy()
    day_energy = pv_days[..., 10:17].sum(axis=-1, keepdims=True)
    moved = day_energy * share * shift_frac
    out[..., 10:17] -= pv_days[..., 10:17] / np.maximum(day_energy, 1e-9) * moved
    out[..., 18:24] += moved / 6.0
    return out


@dataclass
class EnvConfig:
    horizon_years: int = 10
    annual_budget_mw: float = 0.25       # D11: offered subsidized capacity per year
    w_peak: float = 1.0
    w_tr: float = 0.5
    w_v: float = 0.5
    delta_shaping: float = 0.0           # proposal's delta*sum(CF_i*b_i); ablation only
    overload_threshold: float = 1.0      # transformer |S|/cap above this = overload
    bess_share: float = 0.0              # OAT sensitivity (D14)
    bess_shift_frac: float = 0.30
    impedance_scale: float = 0.5         # 0.5 stiff urban (D15); 1.0 weak feeder
    potential_scale: float = 1.0         # scales rooftop potential (D10 sensitivity)
    static_cf_obs: bool = False          # ablation E6: freeze CF at t=0 in the obs
    allow_reserve: bool = True
    install_cost: float = 0.0            # E8 cost-aware variant: lambda * realized/B
    refund_unaccepted: bool = False      # variant: unaccepted / over-headroom offers return to the reserve
    oltc_vmax: float | None = None       # substation OLTC with LDC, head voltage cap (pu)
    oltc_vset: float = 1.0               # LDC setpoint (load-centre voltage, pu)
    oltc_mode: str = "ldc"               # "ldc": hourly line-drop compensation; "fixed": tap held at oltc_vmax
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)


class PlanningCore:
    """Shared simulation core used by both action-space variants."""

    def __init__(self, pv_days: np.ndarray, cfg: EnvConfig, seed: int | None = None):
        self.cfg = cfg
        self.net = Network33(impedance_scale=cfg.impedance_scale, oltc_vmax=cfg.oltc_vmax,
                             oltc_vset=cfg.oltc_vset, oltc_mode=cfg.oltc_mode)
        self.shapes = bus_shapes()                                # (32, 24)
        self.p_nom_kw = self.net.p_load_kw.copy()
        self.q_nom_kvar = self.net.q_load_kvar.copy()
        self.pot_mw = (rho_potential() * self.p_nom_kw / 1000.0
                       * cfg.potential_scale)                     # rooftop potential
        self.pv_days_all = apply_bess(pv_days, cfg.bess_share, cfg.bess_shift_frac)
        self.pv_mean_day = self.pv_days_all.mean(axis=0)          # (24,) nominal PV shape
        self.sampler = ScenarioSampler(self.pv_days_all, cfg.uncertainty, seed)
        self.reset_state()

    def reseed(self, seed: int) -> None:
        self.sampler = ScenarioSampler(self.pv_days_all, self.cfg.uncertainty, seed)

    def reset_state(self) -> None:
        self.installed_mw = np.zeros(N_BUS)
        self.load_scale = 1.0
        self.carryover_mw = 0.0
        self.t = 0
        self._cf0 = self.dynamic_cf()      # frozen CF for the static-CF ablation

    # ------------------------------------------------------------------
    # nominal (noise-free) evaluation used for the observation vector
    # ------------------------------------------------------------------
    def _nominal_net_kw(self) -> np.ndarray:
        gross = self.p_nom_kw[:, None] * self.shapes * self.load_scale    # (32,24)
        pv = self.installed_mw[:, None] * 1000.0 * self.pv_mean_day[None, :]
        return gross - pv

    def dynamic_cf(self) -> np.ndarray:
        """Marginal local peak reduction per MW at the current state (D06)."""
        net = self._nominal_net_kw()
        peak_now = net.max(axis=1)
        probe = net - CF_EPS_MW * 1000.0 * self.pv_mean_day[None, :]
        peak_probe = probe.max(axis=1)
        return (peak_now - peak_probe) / (CF_EPS_MW * 1000.0)

    def observation(self) -> np.ndarray:
        net = self._nominal_net_kw()
        q_net = self.q_nom_kvar[:, None] * self.shapes * self.load_scale
        loading = self.net.transformer_loading(net, q_net).max(axis=1)
        cf = self._cf0 if self.cfg.static_cf_obs else self.dynamic_cf()
        obs = np.concatenate([
            self.installed_mw / self.pot_mw,                        # PV penetration
            np.clip(1.0 - loading, -1.0, 1.0),                      # transformer margin
            np.clip(cf, 0.0, 1.0),                                  # CF (D06 / ablation)
            net.max(axis=1) / self.p_nom_kw,                        # normalized net peak
            [self.carryover_mw / max(self.cfg.annual_budget_mw, 1e-9),
             self.t / self.cfg.horizon_years],
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # one annual transition
    # ------------------------------------------------------------------
    def apply_year(self, offered_mw: np.ndarray, reserve_mw: float) -> dict[str, Any]:
        cfg = self.cfg
        # 1) adoption uncertainty -> realized installations (capped by potential)
        accept = self.sampler.acceptance(
            N_BUS, penetration=self.installed_mw / self.pot_mw)
        headroom = np.maximum(self.pot_mw - self.installed_mw, 0.0)
        realized = np.minimum(offered_mw * accept, headroom)
        self.installed_mw = self.installed_mw + realized
        refund = float(offered_mw.sum() - realized.sum()) if cfg.refund_unaccepted else 0.0
        self.carryover_mw = float(reserve_mw) + refund if cfg.allow_reserve else 0.0

        # 2) demand growth
        growth = self.sampler.demand_growth()
        self.load_scale *= 1.0 + growth

        # 3) stochastic physical evaluation over representative summer days
        noise = self.sampler.load_noise(N_BUS)                       # (32, D, 24)
        rep = self.sampler.rep_days()                                # (D, 24)
        gross = (self.p_nom_kw[:, None, None] * self.shapes[:, None, :]
                 * self.load_scale * noise)                          # (32, D, 24) kW
        q_net = (self.q_nom_kvar[:, None, None] * self.shapes[:, None, :]
                 * self.load_scale * noise)
        pv = self.installed_mw[:, None, None] * 1000.0 * rep[None, :, :]
        net = gross - pv

        n_days = rep.shape[0]
        flat = net.reshape(N_BUS, -1)
        q_flat = q_net.reshape(N_BUS, -1)

        sys_gross = gross.sum(axis=0)                                # (D, 24)
        sys_net = net.sum(axis=0)
        peak_gross = sys_gross.max(axis=1).mean()                    # expected daily peak
        peak_net = sys_net.max(axis=1).mean()
        dpeak_frac = (peak_gross - peak_net) / peak_gross

        loading = self.net.transformer_loading(flat, q_flat)
        overload_frac = float((loading > cfg.overload_threshold).mean())
        v = self.net.voltages_pu(flat, q_flat)
        volt_frac = float(((v < V_MIN_PU) | (v > V_MAX_PU)).mean())
        head = sys_net.min()                                         # most negative import
        rpf_mw = float(max(0.0, -head) / 1000.0)

        # Counterfactual grid stress WITHOUT PV on the same stochastic draws
        # (decision D16): penalties are the PV-attributable *differences*, so
        # load-growth-driven stress the agent cannot influence cancels out.
        flat_g = gross.reshape(N_BUS, -1)
        over_gross = float(
            (self.net.transformer_loading(flat_g, q_flat) > cfg.overload_threshold).mean())
        v_g = self.net.voltages_pu(flat_g, q_flat)
        volt_gross = float(((v_g < V_MIN_PU) | (v_g > V_MAX_PU)).mean())
        over_delta = overload_frac - over_gross      # >0: PV made loading worse
        volt_delta = volt_frac - volt_gross          # <0: PV relieved voltage sag
        # Reporting-only voltage metrics: undervoltage deficit
        # (mean of max(0, V_min - V) over bus-hours, pu) and counts at other
        # thresholds; the reward keeps the count below V_MIN_PU.
        extra = {}
        for tag, vv in (("net", v), ("gross", v_g)):
            extra[f"uv_deficit_{tag}"] = float(np.maximum(V_MIN_PU - vv, 0.0).mean())
            extra[f"uv094_{tag}"] = float((vv < 0.94).mean())
            extra[f"uv096_{tag}"] = float((vv < 0.96).mean())
        if self.net.oltc_vmax is not None:
            p_head = flat.sum(axis=0) / 1000.0
            v0 = np.broadcast_to(self.net.oltc_v0(p_head, q_flat.sum(axis=0) / 1000.0),
                                 p_head.shape).reshape(n_days, 24)
            extra["tap_ops_per_day"] = float((np.abs(np.diff(v0, axis=1)) > 1e-9).sum() / n_days)

        reward = (cfg.w_peak * dpeak_frac
                  - cfg.w_tr * over_delta
                  - cfg.w_v * volt_delta)
        if cfg.install_cost:
            # E8: subsidy outlay is paid on realized (accepted) capacity only
            reward -= cfg.install_cost * float(realized.sum()) / max(
                cfg.annual_budget_mw, 1e-9)
        if cfg.delta_shaping:
            cf = np.clip(self.dynamic_cf(), 0.0, 1.0)
            reward += cfg.delta_shaping * float(cf @ offered_mw) / max(
                cfg.annual_budget_mw, 1e-9)

        self.t += 1
        return {
            "reward": float(reward),
            "dpeak_frac": float(dpeak_frac),
            "dpeak_mw": float((peak_gross - peak_net) / 1000.0),
            "peak_gross_mw": float(peak_gross / 1000.0),
            "peak_net_mw": float(peak_net / 1000.0),
            "overload_frac": overload_frac,
            "overload_delta": over_delta,
            "volt_violation_frac": volt_frac,
            "volt_delta": volt_delta,
            "volt_gross_frac": volt_gross,
            **extra,
            "reverse_flow_mw": rpf_mw,
            "offered_mw": float(offered_mw.sum()),
            "realized_mw": float(realized.sum()),
            "installed_total_mw": float(self.installed_mw.sum()),
            "acceptance_mean": float(accept.mean()),
            "growth": growth,
        }


class PVDeploymentEnv(gym.Env):
    """Continuous-allocation variant (PPO / SAC). Action: raw logits (33,)."""

    metadata = {"render_modes": []}

    def __init__(self, pv_days: np.ndarray, cfg: EnvConfig | None = None,
                 seed: int | None = None):
        self.cfg = cfg or EnvConfig()
        self.core = PlanningCore(pv_days, self.cfg, seed)
        n_act = N_BUS + (1 if self.cfg.allow_reserve else 0)
        self.action_space = spaces.Box(-5.0, 5.0, shape=(n_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(N_BUS * OBS_PER_BUS + 2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "scenario_seed" in options:
            self.core.reseed(int(options["scenario_seed"]))
        elif seed is not None:
            self.core.reseed(seed)
        self.core.reset_state()
        return self.core.observation(), {}

    def step(self, action: np.ndarray):
        shares = softmax(np.asarray(action, dtype=np.float64))
        budget = self.cfg.annual_budget_mw + self.core.carryover_mw
        if self.cfg.allow_reserve:
            reserve = shares[0] * budget
            offered = shares[1:] * budget
        else:
            reserve = 0.0
            offered = shares * budget
        info = self.core.apply_year(offered, reserve)
        terminated = self.core.t >= self.cfg.horizon_years
        return (self.core.observation(), info["reward"], terminated, False, info)


class TranchePVDeploymentEnv(gym.Env):
    """Sequential-tranche variant (DQN), decision D07.

    The annual budget (plus carryover) is divided into `n_tranches` equal
    tranches. Each sub-step, the agent routes one tranche to a neighborhood
    (actions 1..32) or to the reserve (action 0). After the last tranche the
    annual transition runs and the (delayed) annual reward is returned.
    Observation = base observation + fraction of tranches already placed.
    """

    metadata = {"render_modes": []}

    def __init__(self, pv_days: np.ndarray, cfg: EnvConfig | None = None,
                 seed: int | None = None, n_tranches: int = 8):
        self.cfg = cfg or EnvConfig()
        self.n_tranches = n_tranches
        self.core = PlanningCore(pv_days, self.cfg, seed)
        self.action_space = spaces.Discrete(N_BUS + 1)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(N_BUS * OBS_PER_BUS + 3,), dtype=np.float32)
        self._pending = np.zeros(N_BUS)
        self._reserve = 0.0
        self._k = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [self.core.observation(), [self._k / self.n_tranches]]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "scenario_seed" in options:
            self.core.reseed(int(options["scenario_seed"]))
        elif seed is not None:
            self.core.reseed(seed)
        self.core.reset_state()
        self._pending[:] = 0.0
        self._reserve = 0.0
        self._k = 0
        return self._obs(), {}

    def step(self, action: int):
        budget = self.cfg.annual_budget_mw + self.core.carryover_mw
        tranche = budget / self.n_tranches
        if action == 0 and self.cfg.allow_reserve:
            self._reserve += tranche
        else:
            bus = int(action) - 1 if self.cfg.allow_reserve else int(action)
            bus = max(0, min(N_BUS - 1, bus))
            self._pending[bus] += tranche
        self._k += 1

        if self._k < self.n_tranches:
            return self._obs(), 0.0, False, False, {}

        info = self.core.apply_year(self._pending.copy(), self._reserve)
        self._pending[:] = 0.0
        self._reserve = 0.0
        self._k = 0
        terminated = self.core.t >= self.cfg.horizon_years
        return self._obs(), info["reward"], terminated, False, info


def hybrid_offers(core: PlanningCore, action: np.ndarray, s: np.ndarray,
                  prior_kappa: float, logit_scale: float) -> np.ndarray:
    """Offers (MW) of the hybrid policy for priority action u in [-1, 1]^32."""
    a, b = acceptance_params(core)
    head = np.maximum(core.pot_mw - core.installed_mw, 0.0)
    weight = (1.0 + prior_kappa * s) * np.exp(logit_scale * np.clip(action, -1.0, 1.0))
    budget = core.cfg.annual_budget_mw + core.carryover_mw
    return expected_capacity_allocation(a, b, head, budget, weight)


class HybridPVDeploymentEnv(PVDeploymentEnv):
    """Hybrid variant: the agent sets priorities, an analytic layer allocates.

    Action: u in [-1, 1]^32. Priority weights w_i = prior_i * exp(scale * u_i),
    prior_i = 1 + prior_kappa * s_i (s_i: undervoltage relief per MW). Offers
    solve max sum_i w_i E[min(o_i xi_i, H_i)] s.t. sum_i o_i = budget, so the
    whole budget is offered and no offer exceeds what acceptance can absorb.
    Observation: base observation + s_i (32), which makes the voltage state
    observable. With u = 0 and prior_kappa = kappa the policy equals the
    max-exp-cap-V rule.
    """

    def __init__(self, pv_days: np.ndarray, cfg: EnvConfig | None = None,
                 seed: int | None = None, prior_kappa: float = 0.0,
                 logit_scale: float = 2.0):
        super().__init__(pv_days, cfg, seed)
        self.prior_kappa, self.logit_scale = prior_kappa, logit_scale
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_BUS,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(N_BUS * (OBS_PER_BUS + 1) + 2,), dtype=np.float32)
        self._s = np.zeros(N_BUS)

    def _obs(self) -> np.ndarray:
        self._s = voltage_relief_weights(self.core)
        return np.concatenate([self.core.observation(), self._s]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        offered = hybrid_offers(self.core, np.asarray(action, dtype=np.float64),
                                self._s, self.prior_kappa, self.logit_scale)
        info = self.core.apply_year(offered, 0.0)
        terminated = self.core.t >= self.cfg.horizon_years
        return self._obs(), info["reward"], terminated, False, info
