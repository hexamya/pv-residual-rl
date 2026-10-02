"""Stochastic scenario models: adoption, demand growth, irradiance days, load noise.

Decision D09. A ScenarioSampler owns a seeded RNG so that (a) training draws are
reproducible per seed and (b) the evaluation protocol can hand *identical*
scenario sequences to every policy (paired comparisons, decision D13).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import profiles


@dataclass
class UncertaintyConfig:
    growth_mean: float = 0.025          # annual demand growth (Iran urban, ~2-3%)
    growth_std: float = 0.010
    growth_min: float = 0.0
    growth_max: float = 0.06
    load_noise_std: float = 0.05        # multiplicative hourly load noise
    n_rep_days: int = 10                # stochastic representative summer days/year
    accept_scale: float = 1.0           # sensitivity knob on acceptance means
    accept_class_scale: dict | None = None   # per-class factors on acceptance means
    adoption_model: str = "iid"         # "iid" (D09) | "bass" (imitation dynamics)
    bass_p_scale: float = 0.6           # innovation share of the base mean
    bass_q: float = 0.5                 # imitation coefficient (x penetration)


@dataclass
class ScenarioSampler:
    pv_days: np.ndarray                 # (n_days, 24) per-kWp real summer days (PVGIS)
    cfg: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        a, b = profiles.acceptance_beta_params()
        # accept_scale shifts the mean while keeping concentration (D09/OAT).
        mean = a / (a + b) * self.cfg.accept_scale
        if self.cfg.accept_class_scale:
            f = np.array([self.cfg.accept_class_scale.get(c, 1.0) for c in profiles.class_labels()])
            mean = mean * f[:len(mean)]
        mean = np.clip(mean, 0.02, 0.98)
        conc = a + b
        self._acc_a, self._acc_b = mean * conc, (1 - mean) * conc

    # -- annual draws ------------------------------------------------------
    def acceptance(self, n_bus: int,
                   penetration: np.ndarray | None = None) -> np.ndarray:
        """Per-neighborhood subsidy acceptance rate for one period.

        iid mode: Beta(a_i, b_i) (D09). bass mode: the Beta mean follows a
        Bass-style imitation curve, mean_i = p_i + q * penetration_i, so early
        acceptance is lower and grows with the neighborhood's adopter share —
        seeding a neighborhood raises its future acceptance (ablation D17).
        """
        a, b = self._acc_a[:n_bus], self._acc_b[:n_bus]
        if self.cfg.adoption_model == "bass" and penetration is not None:
            base_mean = a / (a + b)
            conc = a + b
            mean = np.clip(base_mean * self.cfg.bass_p_scale
                           + self.cfg.bass_q * penetration[:n_bus], 0.02, 0.98)
            a, b = mean * conc, (1 - mean) * conc
        return self.rng.beta(a, b)

    def demand_growth(self) -> float:
        g = self.rng.normal(self.cfg.growth_mean, self.cfg.growth_std)
        return float(np.clip(g, self.cfg.growth_min, self.cfg.growth_max))

    def rep_days(self) -> np.ndarray:
        """(n_rep_days, 24) bootstrap sample of real PV summer days (kW/kWp)."""
        idx = self.rng.integers(0, self.pv_days.shape[0], size=self.cfg.n_rep_days)
        return self.pv_days[idx]

    def load_noise(self, n_bus: int) -> np.ndarray:
        """(n_bus, n_rep_days, 24) multiplicative load noise."""
        return self.rng.normal(
            1.0, self.cfg.load_noise_std, size=(n_bus, self.cfg.n_rep_days, 24)
        ).clip(0.7, 1.3)
