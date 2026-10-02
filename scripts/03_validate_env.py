"""E1 — Environment validation (thesis experiment 1).

Checks, with figures for the implementation chapter:
  1. Load calibration: system gross peak falls in the afternoon (Iran summer);
     evening ridge ratio reported (bounds the max achievable peak reduction).
  2. CF heterogeneity across neighborhood classes and CF erosion vs penetration
     (dynamic-CF correction D06; duck-curve emergence).
  3. Gymnasium API conformity + step-rate benchmark (feasibility of 1M steps).
  4. LinDistFlow surrogate vs OpenDSS exact solution (decision D02).

Run:  .venv\\Scripts\\python.exe scripts\\03_validate_env.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pvplan import pvgis
from pvplan.environment import EnvConfig, PVDeploymentEnv, TranchePVDeploymentEnv
from pvplan.network import Network33
from pvplan.plotstyle import CLASS_COLORS, GRAY, PALETTE, apply_style, save_fig
from pvplan.profiles import bus_shapes, class_labels, rho_potential

RESULTS = Path(__file__).resolve().parents[1] / "results"
report: dict = {}

apply_style()
net = Network33()
shapes = bus_shapes()
labels = class_labels()
pv_days = pvgis.load_summer_days()
pv_mean = pv_days.mean(axis=0)
hours = np.arange(24)

# ---------------------------------------------------------------- 1. load calibration
gross_sys = (net.p_load_kw[:, None] * shapes).sum(axis=0)          # kW, (24,)
peak_hour = int(gross_sys.argmax())
afternoon_peak = gross_sys[12:18].max()
evening_ridge = gross_sys[19:24].max()
ridge_ratio = evening_ridge / gross_sys.max()
report["system_peak_hour"] = peak_hour
report["system_peak_mw"] = round(gross_sys.max() / 1000, 3)
report["evening_ridge_ratio"] = round(float(ridge_ratio), 4)
assert 12 <= peak_hour <= 17, f"system peak must be afternoon, got {peak_hour}"

# feeder voltage at nominal design load must respect the -5% criterion (D15)
p_nom_hourly = net.p_load_kw[:, None] * shapes
q_nom_hourly = net.q_load_kvar[:, None] * shapes
v_nom_min = net.voltages_pu(p_nom_hourly, q_nom_hourly).min()
report["v_min_nominal_pu"] = round(float(v_nom_min), 4)
assert v_nom_min > 0.945, f"feeder violates planning criterion at design load: {v_nom_min}"

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(hours, gross_sys / 1000, color=PALETTE[0])
ax.annotate(f"system peak {gross_sys.max()/1000:.2f} MW @ {peak_hour}:00",
            xy=(peak_hour, gross_sys.max() / 1000),
            xytext=(peak_hour - 9, gross_sys.max() / 1000 * 1.01), fontsize=9)
ax.annotate(f"evening ridge {evening_ridge/1000:.2f} MW",
            xy=(int(19 + gross_sys[19:24].argmax()), evening_ridge / 1000),
            xytext=(13.5, evening_ridge / 1000 * 0.90), fontsize=9)
ax.set_xlabel("hour (local)")
ax.set_ylabel("system load (MW)")
ax.set_title("Calibrated summer-weekday system load, IEEE 33-bus neighborhoods")
save_fig(fig, "01_system_load_curve", "validation")

# per-class normalized shapes
fig, ax = plt.subplots(figsize=(7, 3.6))
for cls in CLASS_COLORS:
    idx = [i for i, c in enumerate(labels) if c == cls]
    mean_shape = shapes[idx].mean(axis=0)
    ax.plot(hours, mean_shape, color=CLASS_COLORS[cls])
    h_lab = int(np.argmax(mean_shape))
    ax.annotate(cls, xy=(h_lab, mean_shape[h_lab] + 0.015),
                color=CLASS_COLORS[cls], fontsize=9, ha="center")
ax.plot(hours, pv_mean / pv_mean.max(), color=GRAY, linestyle="--")
ax.annotate("PV (normalized)", xy=(8.2, 0.75), color=GRAY, fontsize=9)
ax.set_xlabel("hour (local)")
ax.set_ylabel("normalized load")
ax.set_title("Neighborhood-class load archetypes vs PV production shape")
save_fig(fig, "02_class_archetypes", "validation")

# ------------------------------------------------- 2. dynamic CF + duck curves
pot_mw = rho_potential() * net.p_load_kw / 1000.0
report["total_potential_mw"] = round(float(pot_mw.sum()), 3)

pen_grid = np.linspace(0, 1, 21)
cf_by_class = {c: [] for c in CLASS_COLORS}
eps_kw = 50.0
for pen in pen_grid:
    inst_kw = pen * pot_mw * 1000.0
    netload = net.p_load_kw[:, None] * shapes - inst_kw[:, None] * pv_mean[None, :]
    peak_now = netload.max(axis=1)
    peak_probe = (netload - eps_kw * pv_mean[None, :]).max(axis=1)
    cf = (peak_now - peak_probe) / eps_kw
    for cls in CLASS_COLORS:
        idx = [i for i, c in enumerate(labels) if c == cls]
        cf_by_class[cls].append(cf[idx].mean())

fig, ax = plt.subplots(figsize=(7, 3.6))
for cls, vals in cf_by_class.items():
    ax.plot(pen_grid * 100, vals, color=CLASS_COLORS[cls])
    ax.annotate(cls, xy=(pen_grid[-1] * 100 + 1, vals[-1]), color=CLASS_COLORS[cls],
                fontsize=9, va="center")
ax.set_xlabel("PV penetration (% of rooftop potential)")
ax.set_ylabel("marginal coincidence factor")
ax.set_title("CF erosion with penetration — why static CF is invalid (D06)")
ax.set_xlim(0, 118)
save_fig(fig, "03_cf_erosion", "validation")

report["cf_at_zero"] = {c: round(float(v[0]), 3) for c, v in cf_by_class.items()}
report["cf_at_full"] = {c: round(float(v[-1]), 3) for c, v in cf_by_class.items()}
# Physics found (kept as thesis findings, asserted here):
#  - residential local CF ~ 0 at all penetrations (evening local peak, no PV output);
#  - CF ordering at zero penetration: admin/commercial > mixed > residential;
#  - CF of daytime-peaking classes erodes toward 0 as penetration grows (duck curve).
assert cf_by_class["residential"][0] < 0.05, "residential local CF should be ~0"
assert cf_by_class["commercial"][0] > cf_by_class["mixed"][0] > cf_by_class["residential"][0]
assert cf_by_class["commercial"][-1] < 0.3 * cf_by_class["commercial"][0], \
    "commercial CF must erode with penetration"

# duck curves
fig, ax = plt.subplots(figsize=(7, 3.6))
pens = [0.0, 0.25, 0.5, 0.75, 1.0]
seq = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]  # single-hue ramp
for pen, col in zip(pens, seq):
    inst_kw = pen * pot_mw * 1000.0
    netload = (net.p_load_kw[:, None] * shapes
               - inst_kw[:, None] * pv_mean[None, :]).sum(axis=0)
    ax.plot(hours, netload / 1000, color=col)
    if pen in (0.0, 1.0):
        ax.annotate(f"{int(pen*100)}% potential", xy=(11, netload[11] / 1000),
                    color=col, fontsize=9, va="bottom" if pen == 0 else "top")
ax.set_xlabel("hour (local)")
ax.set_ylabel("system net load (MW)")
ax.set_title("Duck-curve emergence: system net load vs PV penetration")
save_fig(fig, "04_duck_curves", "validation")

# max achievable system peak reduction sweep
dpeaks = []
for pen in pen_grid:
    inst_kw = pen * pot_mw * 1000.0
    netload = (net.p_load_kw[:, None] * shapes
               - inst_kw[:, None] * pv_mean[None, :]).sum(axis=0)
    dpeaks.append((gross_sys.max() - netload.max()) / gross_sys.max() * 100)
report["max_system_dpeak_pct_uniform"] = round(float(max(dpeaks)), 2)

# ------------------------------------------------- 3. env API + speed
cfg = EnvConfig()
env = PVDeploymentEnv(pv_days, cfg, seed=0)
obs, _ = env.reset(seed=0)
assert obs.shape == env.observation_space.shape
t0 = time.perf_counter()
n_steps = 0
for ep in range(60):
    env.reset(seed=ep)
    done = False
    while not done:
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        n_steps += 1
rate = n_steps / (time.perf_counter() - t0)
report["env_steps_per_sec"] = int(rate)
report["random_policy_last_info"] = {
    k: round(v, 4) for k, v in info.items() if isinstance(v, float)}
assert rate > 200, f"env too slow for 1M steps: {rate:.0f} steps/s"

tenv = TranchePVDeploymentEnv(pv_days, cfg, seed=0, n_tranches=8)
obs, _ = tenv.reset(seed=0)
for _ in range(85):
    obs, r, done, trunc, info = tenv.step(int(tenv.action_space.sample()))
    if done:
        obs, _ = tenv.reset()
report["tranche_env_ok"] = True

# ------------------------------------------------- 4. OpenDSS cross-validation
from pvplan.opendss_validate import run_validation

dss_stats = run_validation(n_samples=200, seed=1)
report["opendss_validation"] = {k: round(v, 5) for k, v in dss_stats.items()}
assert dss_stats["v_err_p95_pu"] < 0.01, "LinDistFlow voltage error too large"

RESULTS.mkdir(exist_ok=True)
with open(RESULTS / "validation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print("\nE1 validation PASSED — figures in results/figures/validation/")
