"""Load archetypes, neighborhood classes, rooftop potential and adoption parameters.

Correction D04 to the proposal: instead of one uniform 65/25/10 mix (which makes
every feeder evening-peaking, so rooftop PV without storage could reduce no peak
and the spatial allocation problem degenerates), each neighborhood belongs to one
of four classes with a distinct residential/commercial/administrative mix. Summer
archetypes include the cooling (A/C) component so the *system* coincident peak
falls in the afternoon (Iran's summer system peak occurs ~13:00-17:00), while
residential-heavy feeders keep a dominant evening peak. This produces genuinely
heterogeneous, penetration-dependent coincidence factors across neighborhoods.

All shapes are normalized summer-weekday profiles (24 hourly values, local time).
A bus's hourly load = nominal_peak_kW * composite_shape(h), where the composite
shape is renormalized to max 1 so the nominal load remains the bus's own peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Summer weekday archetypes (fraction of archetype peak, hour 0..23 local).
# Residential: Tehran-type summer profile — afternoon A/C shoulder plus a
#   dominant late-evening peak (returns home + cooling + lighting).
# Commercial: retail/bazaar — midday peak, secondary early-evening shopping.
# Administrative: Iranian office hours (~7:30-14:30) — late-morning peak.
# ---------------------------------------------------------------------------
RESIDENTIAL = np.array([
    0.62, 0.55, 0.50, 0.47, 0.45, 0.44, 0.46, 0.50,
    0.52, 0.55, 0.60, 0.66, 0.72, 0.79, 0.84, 0.86,
    0.84, 0.81, 0.83, 0.89, 0.95, 1.00, 0.97, 0.80,
])
COMMERCIAL = np.array([
    0.15, 0.13, 0.12, 0.12, 0.12, 0.13, 0.16, 0.22,
    0.35, 0.60, 0.85, 0.95, 1.00, 1.00, 0.95, 0.86,
    0.76, 0.80, 0.85, 0.85, 0.75, 0.55, 0.30, 0.18,
])
ADMINISTRATIVE = np.array([
    0.12, 0.12, 0.12, 0.12, 0.12, 0.13, 0.15, 0.45,
    0.80, 0.95, 1.00, 1.00, 0.98, 0.90, 0.70, 0.40,
    0.25, 0.20, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12,
])

ARCHETYPES = np.stack([RESIDENTIAL, COMMERCIAL, ADMINISTRATIVE])  # (3, 24)


@dataclass(frozen=True)
class NeighborhoodClass:
    name: str
    mix: tuple[float, float, float]      # residential, commercial, administrative
    rho_potential: float                 # rooftop potential / nominal peak (D10)
    accept_mean: float                   # mean subsidy acceptance rate (D09)
    accept_conc: float                   # Beta concentration (a+b); higher = less noisy


# Class parameters. accept_mean is an income/awareness proxy: commercial and
# administrative building owners accept subsidies more readily than the average
# residential household (capital access), per adoption literature (Sunar &
# Swaminathan 2024; Sadabadi & Rahimirad 2025 for Iran).
CLASSES: dict[str, NeighborhoodClass] = {
    "residential": NeighborhoodClass("residential", (0.80, 0.15, 0.05), 0.90, 0.45, 10.0),
    "mixed":       NeighborhoodClass("mixed",       (0.60, 0.30, 0.10), 0.65, 0.55, 10.0),
    "commercial":  NeighborhoodClass("commercial",  (0.35, 0.45, 0.20), 0.45, 0.70, 12.0),
    "admin":       NeighborhoodClass("admin",       (0.30, 0.30, 0.40), 0.35, 0.65, 12.0),
}

# Deterministic class assignment for the 32 load buses (standard buses 2..33).
# Pattern (seeded choice, fixed for reproducibility): the main trunk near the
# substation and the heavy-load lateral (buses 23-25) are commercial/mixed;
# feeder ends are residential; a small administrative pocket mid-feeder.
# Counts: residential 12, mixed 10, commercial 7, admin 3.
CLASS_ASSIGNMENT: list[str] = [
    # bus:    2             3             4             5
    "commercial", "commercial", "mixed",      "mixed",
    # bus:    6             7             8             9
    "mixed",      "admin",      "mixed",      "residential",
    # bus:   10            11            12            13
    "residential", "residential", "mixed",     "residential",
    # bus:   14            15            16            17
    "residential", "residential", "residential", "residential",
    # bus:   18            19            20            21
    "residential", "mixed",      "residential", "residential",
    # bus:   22            23            24            25
    "residential", "commercial", "commercial", "commercial",
    # bus:   26            27            28            29
    "mixed",      "admin",      "mixed",      "mixed",
    # bus:   30            31            32            33
    "commercial", "mixed",      "admin",      "residential",
]

# Reactive power of PV systems: modern rooftop inverters operate near unity
# power factor; net Q is unchanged by PV (conservative for voltage rise).
# Load power factor per bus is implied by the IEEE 33-bus P/Q data.


def bus_shapes() -> np.ndarray:
    """(32, 24) composite normalized shape per load bus (own peak = 1)."""
    shapes = []
    for cls_name in CLASS_ASSIGNMENT:
        mix = np.array(CLASSES[cls_name].mix)
        composite = mix @ ARCHETYPES
        shapes.append(composite / composite.max())
    return np.array(shapes)


def rho_potential() -> np.ndarray:
    """(32,) rooftop potential as a multiple of nominal bus peak (D10)."""
    return np.array([CLASSES[c].rho_potential for c in CLASS_ASSIGNMENT])


def acceptance_beta_params() -> tuple[np.ndarray, np.ndarray]:
    """(a, b) arrays of per-bus Beta parameters for subsidy acceptance (D09)."""
    means = np.array([CLASSES[c].accept_mean for c in CLASS_ASSIGNMENT])
    conc = np.array([CLASSES[c].accept_conc for c in CLASS_ASSIGNMENT])
    return means * conc, (1.0 - means) * conc


def class_labels() -> list[str]:
    return list(CLASS_ASSIGNMENT)
