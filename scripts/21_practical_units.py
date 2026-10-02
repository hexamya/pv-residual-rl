"""Express returns in engineering units (kW of peak, bus-hours of undervoltage).

The voltage term of the return is -0.5 * sum_t (V_net - V_gross), with V the
fraction of bus-hours outside 0.95-1.05 pu on the D sampled summer days. The
D days stand for a 92-day summer, so the PV-attributable change in violating
bus-hours per summer is -volt_delta_sum / T * (32 * 24 * 92). A no-PV run on
the same scenarios gives the gross peak (MW) and gross violations, so the
peak term converts to kW and relieved violations to a share of all violations.
Mean-annual kW is approximate (sum_t dP_t/P_t * mean_t P_t); final-year kW is
exact.

Usage: python scripts/21_practical_units.py
Outputs: results/units_weak.csv, units_stiff.csv, units_weak_pairs.csv,
         units_robust_pairs.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pvplan import pvgis
from pvplan.environment import N_BUS, EnvConfig, PlanningCore
from pvplan.stats import bootstrap_ci

RESULTS = ROOT / "results"
SEEDS = [10_000 + k for k in range(100)]
BUS_HOURS_PER_SUMMER = N_BUS * 24 * 92


def no_pv_reference(cfg: EnvConfig) -> pd.DataFrame:
    """Per-scenario gross peak and gross violation share, without any PV."""
    core = PlanningCore(pvgis.load_summer_days(), cfg)
    rows = []
    for s in SEEDS:
        core.reseed(s)
        core.reset_state()
        peaks, viol = [], []
        for _ in range(cfg.horizon_years):
            info = core.apply_year(np.zeros(N_BUS), 0.0)
            peaks.append(info["peak_gross_mw"])
            viol.append(info["volt_violation_frac"])
        rows.append({"scenario_seed": s, "peak_gross_mean_mw": np.mean(peaks),
                     "viol_gross_sum": np.sum(viol)})
    return pd.DataFrame(rows)


def family_means(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    return df.groupby(["family", "scenario_seed"]).mean(numeric_only=True).reset_index()


def to_units(comp: pd.DataFrame, ref: pd.DataFrame, T: int) -> pd.DataFrame:
    c = comp.merge(ref, on="scenario_seed")
    c["v_term"] = -0.5 * c["volt_delta_sum"]
    c["peak_term"] = c["return_sum"] - c["v_term"] + 0.5 * c["overload_delta_sum"]
    c["peak_kw_mean_annual"] = 1000 * c["peak_term"] / T * c["peak_gross_mean_mw"]
    c["peak_kw_final_year"] = 1000 * c["dpeak_mw_final"]
    c["uv_bus_hours_relieved_per_summer"] = (-c["volt_delta_sum"] / T
                                             * BUS_HOURS_PER_SUMMER)
    c["uv_share_relieved_pct"] = 100 * -c["volt_delta_sum"] / c["viol_gross_sum"]
    c["uv_share_remaining_pct"] = 100 * (c["viol_gross_sum"] + c["volt_delta_sum"]) / T
    return c


def paired(c: pd.DataFrame, a: str, b: str, col: str) -> dict:
    x = c[c.family == a].set_index("scenario_seed")[col]
    y = c[c.family == b].set_index("scenario_seed")[col].loc[x.index]
    d = (x - y).values
    lo, hi = bootstrap_ci(d)
    return {"policy": a, "reference": b, "metric": col, "diff": d.mean(),
            "ci_lo": lo, "ci_hi": hi}


COLS = ["return_sum", "peak_kw_mean_annual", "peak_kw_final_year",
        "uv_bus_hours_relieved_per_summer", "uv_share_relieved_pct",
        "uv_share_remaining_pct", "installed_total_mw"]


def main() -> None:
    pd.set_option("display.width", 220)

    # weak feeder (main Path A case)
    weak_ref = no_pv_reference(EnvConfig(impedance_scale=1.0))
    weak = to_units(family_means(pd.read_csv(RESULTS / "hybrid_evaluation.csv")),
                    weak_ref, 10)
    tw = weak.groupby("family")[COLS].mean().sort_values("return_sum", ascending=False)
    tw.to_csv(RESULTS / "units_weak.csv")
    pairs = pd.DataFrame([paired(weak, a, b, m)
                          for a, b in [("residual", "max-exp-cap-V"),
                                       ("residual", "rolling-lp-V"), ("residual", "ppo"),
                                       ("max-exp-cap-V", "max-exp-cap")]
                          for m in ["peak_kw_mean_annual",
                                    "uv_bus_hours_relieved_per_summer"]])
    pairs.to_csv(RESULTS / "units_weak_pairs.csv", index=False)
    print(f"weak feeder: gross peak {weak_ref.peak_gross_mean_mw.mean():.3f} MW, "
          f"undervoltage share without PV "
          f"{100 * weak_ref.viol_gross_sum.mean() / 10:.1f}% of bus-hours")
    print(tw.round(2).to_string(), "\n\n", pairs.round(2).to_string(index=False))

    # stiff feeder (paper's nominal case): peak in kW
    stiff_ref = no_pv_reference(EnvConfig())
    nom = pd.read_csv(RESULTS / "evaluation_nominal.csv")
    nom = pd.concat([nom, pd.read_csv(RESULTS / "review_r1_nominal.csv")], ignore_index=True)
    stiff = to_units(family_means(nom), stiff_ref, 10)
    ts = stiff.groupby("family")[["return_sum", "peak_kw_mean_annual",
                                  "peak_kw_final_year", "installed_total_mw"]].mean()
    ts = ts.sort_values("return_sum", ascending=False)
    ts.to_csv(RESULTS / "units_stiff.csv")
    print(f"\nstiff feeder: gross peak {stiff_ref.peak_gross_mean_mw.mean():.3f} MW")
    print(ts.round(2).to_string())
    for a, b in [("max-exp-cap", "ppo"), ("ppo", "rolling-lp")]:
        print(paired(stiff, a, b, "peak_kw_mean_annual"))

    # robustness settings: residual minus rule in undervoltage bus-hours
    rob = pd.read_csv(RESULTS / "robust_evaluation.csv")
    rows = []
    for setting, g in rob.groupby("setting"):
        T = 5 if setting == "horizon_years=5" else 15 if setting == "horizon_years=15" else 10
        c = family_means(g)
        c["uv_bus_hours_relieved_per_summer"] = (-c["volt_delta_sum"] / T
                                                 * BUS_HOURS_PER_SUMMER)
        c["peak_term"] = c["return_sum"] + 0.5 * c["volt_delta_sum"]
        r = paired(c, "residual", "max-exp-cap-V", "uv_bus_hours_relieved_per_summer")
        r["setting"] = setting
        r["rule_uv_bus_hours"] = c[c.family == "max-exp-cap-V"][
            "uv_bus_hours_relieved_per_summer"].mean()
        rows.append(r)
    rp = pd.DataFrame(rows)[["setting", "rule_uv_bus_hours", "diff", "ci_lo", "ci_hi"]]
    rp.to_csv(RESULTS / "units_robust_pairs.csv", index=False)
    print("\nresidual minus rule, undervoltage bus-hours relieved per summer:\n",
          rp.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
