"""PVGIS data loading: hourly per-kWp PV production for Tehran.

Data source (decision D05): PVGIS v5.3 seriescalc API, radiation database
PVGIS-SARAH3 (satellite), meteo ERA5, 2005-2023, for a 1-kWp crystalline-Si
building-mounted system at PVGIS-optimal tilt (32 deg) facing south, 14% system
losses. The returned hourly power P [W per kWp] already includes the
temperature-dependent efficiency term of the proposal's PV equation
(P = eta*A*G*(1 - beta*(T_cell-25))) via JRC's validated performance model.

PVGIS timestamps are UTC; Tehran is UTC+3:30. We shift by 3.5 h and floor to
the local hour (<=30 min alignment error, negligible at annual planning scale).

Primary stochastic model: bootstrap sampling of real summer days (Jun-Aug).
For the thesis text we also fit a Beta distribution to the normalized daily
production index (proposal compatibility).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_JSON = Path(__file__).resolve().parents[2] / "data" / "raw" / "pvgis_tehran_2005_2023.json"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"
SUMMER_MONTHS = (6, 7, 8)


def load_hourly(raw_json: Path = RAW_JSON) -> pd.DataFrame:
    """Parse PVGIS JSON -> DataFrame indexed by local (Tehran) time.

    Columns: p_kw_per_kwp (AC power of 1-kWp system), g_wm2 (POA irradiance),
    t2m (ambient temperature).
    """
    with open(raw_json, encoding="utf-8") as f:
        data = json.load(f)
    rows = data["outputs"]["hourly"]
    df = pd.DataFrame(rows)
    ts_utc = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")
    ts_local = ts_utc + pd.Timedelta(hours=3.5)
    out = pd.DataFrame(
        {
            "p_kw_per_kwp": df["P"].to_numpy() / 1000.0,
            "g_wm2": df["G(i)"].to_numpy(),
            "t2m": df["T2m"].to_numpy(),
        },
        index=ts_local.dt.floor("h"),
    )
    out.index.name = "time_local"
    return out


def summer_day_matrix(df: pd.DataFrame) -> np.ndarray:
    """(n_days, 24) per-kWp production for complete summer days (kW/kWp)."""
    d = df[df.index.month.isin(SUMMER_MONTHS)].copy()
    d["date"] = d.index.date
    d["hour"] = d.index.hour
    pivot = d.pivot_table(index="date", columns="hour", values="p_kw_per_kwp")
    pivot = pivot.dropna()          # keep complete days only
    return pivot.to_numpy()


def build_processed(raw_json: Path = RAW_JSON, out_dir: Path = PROCESSED) -> dict:
    """Create processed artifacts: summer day matrix + Beta fit of daily index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_hourly(raw_json)
    days = summer_day_matrix(df)
    np.save(out_dir / "pv_summer_days.npy", days)

    # Daily production index = daily energy / 95th-percentile daily energy
    # ("clear-sky-like" reference). Beta fit reported for the thesis (D05).
    daily_energy = days.sum(axis=1)
    ref = np.quantile(daily_energy, 0.95)
    index = np.clip(daily_energy / ref, 1e-6, 1 - 1e-6)
    from scipy import stats

    a, b, loc, scale = stats.beta.fit(index, floc=0, fscale=1)
    meta = {
        "n_days": int(days.shape[0]),
        "mean_daily_kwh_per_kwp": float(daily_energy.mean()),
        "p95_daily_kwh_per_kwp": float(ref),
        "beta_a": float(a),
        "beta_b": float(b),
    }
    with open(out_dir / "pv_summer_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_summer_days(out_dir: Path = PROCESSED) -> np.ndarray:
    return np.load(out_dir / "pv_summer_days.npy")
