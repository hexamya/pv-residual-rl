"""Shared matplotlib style for all thesis figures.

Categorical palette: Okabe-Ito (CVD-safe; validated with the six-check palette
validator — CVD separation PASS; amber/pink/light-blue carry direct labels
because of the contrast warning). One axis per chart; recessive grid; direct
labels preferred over dense legends.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# fixed assignment order — never cycled/reshuffled between figures
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
CLASS_COLORS = {
    "residential": "#0072B2",
    "mixed": "#E69F00",
    "commercial": "#009E73",
    "admin": "#D55E00",
}
GRAY = "#8a8a8a"

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e0",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def save_fig(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    out = FIG_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
