"""LinDistFlow vs OpenDSS on both feeders (stiff x0.5 and weak x1.0 impedances).

Same random states as scripts/03_validate_env.py (200 samples, seed 0), plus the
mean bias and the agreement on bus-states below 0.95 pu, which the voltage term
of the reward counts.

Usage: python scripts/23_validate_feeders.py
Output: results/validation_feeders.json
"""

from __future__ import annotations

import json
from pathlib import Path

from pvplan.opendss_validate import run_validation

RESULTS = Path(__file__).resolve().parents[1] / "results"

if __name__ == "__main__":
    out = {name: run_validation(impedance_scale=s)
           for name, s in [("stiff", 0.5), ("weak", 1.0)]}
    (RESULTS / "validation_feeders.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
