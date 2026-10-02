"""IEEE 33-bus distribution network model with LinDistFlow sensitivity matrices.

The IEEE 33-bus radial test feeder (Baran & Wu, 1989) is used as the reference
urban distribution network (proposal section 8-2-1). Each of the 32 load buses
is treated as one "neighborhood"; its nominal load is the diversified coincident
peak demand of that neighborhood.

Power flow during RL training uses the linearized DistFlow (LinDistFlow)
approximation. Because the topology is fixed and radial, the map from bus net
injections to (branch flows, bus voltages) is linear and precomputable:

    P_branch = D  @ p_bus          (D: downstream incidence matrix)
    V_pu     = 1 - Rc @ p_pu - Xc @ q_pu
               (Rc/Xc: common-path resistance/reactance matrices, per unit)

so a full network evaluation over H hours is two matrix products. The exact
nonlinear solution (OpenDSS) is used only for validation (see decision D02).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Standard IEEE 33-bus data (Baran & Wu 1989).
# Columns: from_bus, to_bus, R_ohm, X_ohm, P_kW, Q_kvar   (load at to_bus)
# Buses are 1-indexed; bus 1 is the substation (feeder head).
# ---------------------------------------------------------------------------
BRANCH_DATA: list[tuple[int, int, float, float, float, float]] = [
    (1, 2, 0.0922, 0.0470, 100.0, 60.0),
    (2, 3, 0.4930, 0.2511, 90.0, 40.0),
    (3, 4, 0.3660, 0.1864, 120.0, 80.0),
    (4, 5, 0.3811, 0.1941, 60.0, 30.0),
    (5, 6, 0.8190, 0.7070, 60.0, 20.0),
    (6, 7, 0.1872, 0.6188, 200.0, 100.0),
    (7, 8, 0.7114, 0.2351, 200.0, 100.0),
    (8, 9, 1.0300, 0.7400, 60.0, 20.0),
    (9, 10, 1.0440, 0.7400, 60.0, 20.0),
    (10, 11, 0.1966, 0.0650, 45.0, 30.0),
    (11, 12, 0.3744, 0.1238, 60.0, 35.0),
    (12, 13, 1.4680, 1.1550, 60.0, 35.0),
    (13, 14, 0.5416, 0.7129, 120.0, 80.0),
    (14, 15, 0.5910, 0.5260, 60.0, 10.0),
    (15, 16, 0.7463, 0.5450, 60.0, 20.0),
    (16, 17, 1.2890, 1.7210, 60.0, 20.0),
    (17, 18, 0.7320, 0.5740, 90.0, 40.0),
    (2, 19, 0.1640, 0.1565, 90.0, 40.0),
    (19, 20, 1.5042, 1.3554, 90.0, 40.0),
    (20, 21, 0.4095, 0.4784, 90.0, 40.0),
    (21, 22, 0.7089, 0.9373, 90.0, 40.0),
    (3, 23, 0.4512, 0.3083, 90.0, 50.0),
    (23, 24, 0.8980, 0.7091, 420.0, 200.0),
    (24, 25, 0.8960, 0.7011, 420.0, 200.0),
    (6, 26, 0.2030, 0.1034, 60.0, 25.0),
    (26, 27, 0.2842, 0.1447, 60.0, 25.0),
    (27, 28, 1.0590, 0.9337, 60.0, 20.0),
    (28, 29, 0.8042, 0.7006, 120.0, 70.0),
    (29, 30, 0.5075, 0.2585, 200.0, 600.0),
    (30, 31, 0.9744, 0.9630, 150.0, 70.0),
    (31, 32, 0.3105, 0.3619, 210.0, 100.0),
    (32, 33, 0.3410, 0.5302, 60.0, 40.0),
]

V_BASE_KV = 12.66          # line-to-line base voltage
S_BASE_MVA = 1.0           # power base
Z_BASE_OHM = V_BASE_KV**2 / S_BASE_MVA

# Impedance calibration (decision D15): the classic IEEE 33-bus feeder sags to
# ~0.904 pu at nominal load — typical of a weak overhead rural feeder, not an
# urban 20 kV underground-cable network. Branch impedances are scaled by 0.5 so
# the feeder meets the +/-5% planning criterion at design load, consistent with
# the proposal's calibration to Iranian urban cable feeders.
IMPEDANCE_SCALE = 0.5

# Distribution transformer standard (proposal section 8-2-1, decision D03)
TR_UNIT_KVA = 250.0
TR_DESIGN_LOADING = 0.75   # transformers sized for ~75% loading at nominal peak

V_MIN_PU = 0.95
V_MAX_PU = 1.05

# Substation on-load tap changer: 0.625% steps, line-drop
# compensation to the feeder's electrical load centre (see Network33.oltc_v0).
OLTC_STEP_PU = 0.00625


@dataclass
class Network33:
    """IEEE 33-bus feeder with precomputed linear sensitivity matrices.

    All public arrays are indexed over the 32 *load* buses (bus 2..33 in the
    standard numbering -> indices 0..31 here). Hour-resolved quantities accept
    arrays of shape (n_load, H).
    """

    n_bus: int = 33
    n_load: int = 32
    impedance_scale: float = IMPEDANCE_SCALE   # 0.5 = stiff urban (D15); 1.0 = weak feeder
    oltc_vmax: float | None = None   # None: head held at 1.0 pu; else OLTC+LDC capped here
    oltc_vset: float = 1.0           # LDC setpoint at the load centre
    oltc_mode: str = "ldc"           # "ldc" (hourly line-drop compensation) or "fixed" (tap at oltc_vmax)
    p_load_kw: np.ndarray = field(init=False)     # nominal coincident peak P per neighborhood
    q_load_kvar: np.ndarray = field(init=False)
    tr_capacity_kva: np.ndarray = field(init=False)
    downstream: np.ndarray = field(init=False)    # (n_branch, n_load) 0/1
    r_common_pu: np.ndarray = field(init=False)   # (n_load, n_load)
    x_common_pu: np.ndarray = field(init=False)
    branch_r_ohm: np.ndarray = field(init=False)
    branch_x_ohm: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_branch = len(BRANCH_DATA)
        assert n_branch == self.n_bus - 1

        from_bus = np.array([b[0] for b in BRANCH_DATA])
        to_bus = np.array([b[1] for b in BRANCH_DATA])
        self.branch_r_ohm = np.array([b[2] for b in BRANCH_DATA]) * self.impedance_scale
        self.branch_x_ohm = np.array([b[3] for b in BRANCH_DATA]) * self.impedance_scale
        self.p_load_kw = np.array([b[4] for b in BRANCH_DATA])
        self.q_load_kvar = np.array([b[5] for b in BRANCH_DATA])

        # Path from each bus to the root (list of branch indices).
        parent_branch: dict[int, int] = {}   # bus -> branch index feeding it
        for k, (f, t) in enumerate(zip(from_bus, to_bus)):
            parent_branch[t] = k
        parent_bus = {t: f for f, t in zip(from_bus, to_bus)}

        def path_branches(bus: int) -> list[int]:
            path = []
            while bus != 1:
                k = parent_branch[bus]
                path.append(k)
                bus = parent_bus[bus]
            return path

        # Load bus b (standard number 2..33) -> row index b-2.
        load_buses = to_bus  # each branch's to_bus carries that neighborhood's load
        paths = [set(path_branches(int(b))) for b in load_buses]

        self.downstream = np.zeros((n_branch, self.n_load))
        for j, path in enumerate(paths):
            for k in path:
                self.downstream[k, j] = 1.0

        r_pu = self.branch_r_ohm / Z_BASE_OHM
        x_pu = self.branch_x_ohm / Z_BASE_OHM
        self.r_common_pu = np.zeros((self.n_load, self.n_load))
        self.x_common_pu = np.zeros((self.n_load, self.n_load))
        for i in range(self.n_load):
            for j in range(self.n_load):
                common = paths[i] & paths[j]
                self.r_common_pu[i, j] = sum(r_pu[k] for k in common)
                self.x_common_pu[i, j] = sum(x_pu[k] for k in common)

        # LDC impedance: with loads in their nominal proportions the mean bus
        # voltage drop equals r_ldc * P_head + x_ldc * Q_head, so the OLTC
        # regulates an estimate of the mean bus voltage (the load centre).
        pn, qn = self.p_load_kw / 1000.0, self.q_load_kvar / 1000.0
        self.r_ldc_pu = float((self.r_common_pu @ pn).mean() / pn.sum())
        self.x_ldc_pu = float((self.x_common_pu @ qn).mean() / qn.sum())

        # Transformer bank per neighborhood, sized for the nominal peak.
        s_peak = np.hypot(self.p_load_kw, self.q_load_kvar)
        n_tr = np.maximum(1, np.ceil(s_peak / (TR_DESIGN_LOADING * TR_UNIT_KVA)))
        self.tr_capacity_kva = n_tr * TR_UNIT_KVA

    # ------------------------------------------------------------------
    # Linear power-flow surrogate (vectorized over hours)
    # ------------------------------------------------------------------
    def voltages_pu(self, p_net_kw: np.ndarray, q_net_kvar: np.ndarray) -> np.ndarray:
        """Bus voltage magnitudes (pu) for net load (consumption positive).

        p_net_kw, q_net_kvar: (n_load,) or (n_load, H). PV injection enters as
        negative net load. LinDistFlow: V = V0 - Rc@p_pu - Xc@q_pu, with
        V0 = 1 unless an OLTC is modelled (see oltc_v0).
        """
        p_pu = np.asarray(p_net_kw) / 1000.0 / S_BASE_MVA
        q_pu = np.asarray(q_net_kvar) / 1000.0 / S_BASE_MVA
        return (self.oltc_v0(p_pu.sum(axis=0), q_pu.sum(axis=0))
                - self.r_common_pu @ p_pu - self.x_common_pu @ q_pu)

    def oltc_v0(self, p_head_pu, q_head_pu):
        """Head voltage set by the OLTC for the given head power (per unit).

        Stateless hourly control with line-drop compensation: the tap that
        brings the compensated load-centre voltage closest to oltc_vset,
        limited to [2 - oltc_vmax, oltc_vmax] (the head bus must itself stay
        within limits). Without an OLTC the head is held at 1.0 pu; in "fixed"
        mode the tap is held at oltc_vmax in every hour.
        """
        if self.oltc_vmax is None:
            return 1.0
        if self.oltc_mode == "fixed":
            return self.oltc_vmax
        want = self.oltc_vset + self.r_ldc_pu * p_head_pu + self.x_ldc_pu * q_head_pu
        tap = np.round((want - 1.0) / OLTC_STEP_PU) * OLTC_STEP_PU
        return np.clip(1.0 + tap, 2.0 - self.oltc_vmax, self.oltc_vmax)

    def branch_flows_kw(self, p_net_kw: np.ndarray) -> np.ndarray:
        """Active power flow (kW) on each branch (positive = away from root)."""
        return self.downstream @ np.asarray(p_net_kw)

    def feeder_head_kw(self, p_net_kw: np.ndarray) -> np.ndarray:
        """Net active power drawn from the upstream grid (kW). Negative = reverse flow."""
        return np.asarray(p_net_kw).sum(axis=0)

    def transformer_loading(self, p_net_kw: np.ndarray, q_net_kvar: np.ndarray) -> np.ndarray:
        """|S_net| / capacity per neighborhood; backfeed counts as loading (D03)."""
        s = np.hypot(np.asarray(p_net_kw), np.asarray(q_net_kvar))
        cap = self.tr_capacity_kva if np.asarray(p_net_kw).ndim == 1 else self.tr_capacity_kva[:, None]
        return s / cap
