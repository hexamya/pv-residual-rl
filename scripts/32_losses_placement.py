"""How much does placement change the feeder-head peak once losses are included?

Proposition 1 is exact for the lossless model. In OpenDSS, for a given total
PV capacity, compare the reduction of the feeder-head peak (loads plus losses)
when PV fills the neighbourhoods farthest from the substation first, closest
first, or in 20 random orders. Mid-programme load level (1.025^5), mean PV day,
nominal load shapes.

Usage: python scripts/32_losses_placement.py
Output: results/losses_placement.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pvplan import pvgis
from pvplan.network import Network33
from pvplan.opendss_validate import build_circuit, solve_state
from pvplan.profiles import bus_shapes, rho_potential

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    pv = pvgis.load_summer_days().mean(axis=0)
    shapes = bus_shapes()
    growth = 1.025 ** 5
    rng = np.random.default_rng(0)
    out = {}
    for name, imp in [("stiff", 0.5), ("weak", 1.0)]:
        net = Network33(impedance_scale=imp)
        build_circuit(net)
        pot = rho_potential() * net.p_load_kw / 1000.0
        rc = np.diag(net.r_common_pu)

        def head_peak(x_mw):
            return max(solve_state(net.p_load_kw * shapes[:, h] * growth - x_mw * 1000 * pv[h],
                                   net.q_load_kvar * shapes[:, h] * growth)[1] for h in range(24))

        def fill(order, total):
            x, rem = np.zeros(32), total
            for i in order:
                x[i] = min(pot[i], rem)
                rem -= x[i]
            return x

        base = head_peak(np.zeros(32))
        out[name] = {"no_pv_head_peak_kw": base}
        for total in (0.25, 0.5, 0.75, 1.0, 1.4):
            far = base - head_peak(fill(np.argsort(-rc), total))
            near = base - head_peak(fill(np.argsort(rc), total))
            rnd = [base - head_peak(fill(rng.permutation(32), total)) for _ in range(20)]
            lossless = 1000 * total * pv.max()                 # upper bound if peak stays midday
            out[name][f"{total}MW"] = {"far_first_kw": far, "substation_first_kw": near,
                                       "random_min_kw": min(rnd), "random_max_kw": max(rnd),
                                       "spread_kw": far - near}
        print(name, json.dumps(out[name], indent=1))
    (RESULTS / "losses_placement.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
