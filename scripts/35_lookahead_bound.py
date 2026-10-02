"""A scenario-based lookahead baseline and a perfect-information bound.

rollout: RolloutCapacityPolicy (pvplan.baselines) on the 100 test scenarios.
         Each year it chooses the voltage weight kappa from KAPPAS_C by simulating
         the rest of the horizon with the base rule on 40 futures sampled from the
         model (independent of the evaluated scenario). Settings: the nominal weak
         feeder (base kappa 0.5) and the weak feeder with an OLTC capped at
         1.02 pu (base kappa re-tuned to 1.0, D29).
bound:   wait-and-see bound per test scenario. With the scenario's acceptance,
         growth, irradiance and load noise known in advance, installations a_it
         cost a_it / xi_it of budget, and the ten-year return is maximised exactly
         (LinDistFlow, as in the environment): the peak term through an epigraph
         of the daily peaks, the voltage term with one binary per bus-hour that
         PV could lift above 0.95 pu. Budget may be carried over and new
         violations (overvoltage, backfeed overloads) are ignored, so the value
         bounds the return of every policy, including clairvoyant ones. The MILP
         dual bound at the time limit is reported, together with the best
         solution found (the wait-and-see return, checked by replay).
stats:   paired comparisons and the share of the bound attained.

Usage:
  python scripts/35_lookahead_bound.py rollout [--workers 6]
  python scripts/35_lookahead_bound.py rollout-sens [--workers 6]
  python scripts/35_lookahead_bound.py bound --feeder weak [--time-limit 120] [--workers 6]
  python scripts/35_lookahead_bound.py stats
Outputs: results/rollout_evaluation.csv, rollout_sensitivity.csv, bound_{feeder}.csv, lookahead_stats.csv,
         lookahead_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pvplan.environment import EnvConfig

RESULTS = ROOT / "results"
SEEDS = [10_000 + k for k in range(100)]
KAPPAS_C = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
N_SAMPLES = 40
IMP = {"stiff": 0.5, "weak": 1.0}


def setting_cfg(setting: str) -> EnvConfig:
    return EnvConfig(impedance_scale=1.0, oltc_vmax=1.02 if setting == "weak-oltc102" else None)


def base_kappa(setting: str) -> float:
    if setting == "weak-oltc102":
        return float(json.loads((RESULTS / "oltc_kappa.json").read_text())["1.02"])
    return float(json.loads((RESULTS / "weak_tuned.json").read_text())["kappa"])


# ------------------------------------------------------------------ rollout
def _rollout_job(job: tuple) -> pd.DataFrame:
    from pvplan import pvgis
    from pvplan.baselines import RolloutCapacityPolicy
    from pvplan.evaluation import evaluate_policy
    setting, seed = job
    pol = RolloutCapacityPolicy(KAPPAS_C, base_kappa(setting), N_SAMPLES)
    df = evaluate_policy("rollout", pol, pvgis.load_summer_days(), setting_cfg(setting), [seed])
    df["setting"], df["kappa_path"] = setting, ",".join(f"{k:g}" for k in pol.chosen)
    return df


SENS = {"samples=20": (KAPPAS_C, 20), "samples=80": (KAPPAS_C, 80),
        "fine grid": ([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], N_SAMPLES)}


def _sens_job(job: tuple) -> pd.DataFrame:
    from pvplan import pvgis
    from pvplan.baselines import RolloutCapacityPolicy
    from pvplan.evaluation import evaluate_policy
    variant, seed = job
    grid, n = SENS[variant]
    pol = RolloutCapacityPolicy(grid, base_kappa("weak"), n)
    df = evaluate_policy("rollout", pol, pvgis.load_summer_days(), setting_cfg("weak"), [seed])
    df["variant"], df["kappa_path"] = variant, ",".join(f"{k:g}" for k in pol.chosen)
    return df


def cmd_rollout_sens(workers: int) -> None:
    """Sensitivity of the lookahead rule (nominal weak feeder) to the number of
    sampled futures and to the weight grid."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        df = pd.concat(list(ex.map(_sens_job, [(v, k) for v in SENS for k in SEEDS], chunksize=4)),
                       ignore_index=True)
    df.to_csv(RESULTS / "rollout_sensitivity.csv", index=False)
    print(df.groupby("variant")["return_sum"].mean().round(4).to_string())


def cmd_rollout(workers: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(s, k) for s in ("weak", "weak-oltc102") for k in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        df = pd.concat(list(ex.map(_rollout_job, jobs, chunksize=4)), ignore_index=True)
    df.to_csv(RESULTS / "rollout_evaluation.csv", index=False)
    print(df.groupby("setting")["return_sum"].mean().round(4).to_string())


# ------------------------------------------------------------------ bound
def record_draws(feeder: str, scenario: int) -> list[dict]:
    """Exogenous draws of one test scenario, year by year (identical for all policies)."""
    from pvplan import pvgis
    from pvplan.baselines import MaxExpectedCapacityPolicy
    from pvplan.environment import PVDeploymentEnv
    env = PVDeploymentEnv(pvgis.load_summer_days(), EnvConfig(impedance_scale=IMP[feeder]))
    env.reset(options={"scenario_seed": scenario})
    s, years, cur = env.core.sampler, [], {}
    wrap = {"acceptance": "xi", "demand_growth": "g", "load_noise": "noise", "rep_days": "days"}
    for name, key in wrap.items():
        fn = getattr(s, name)
        setattr(s, name, (lambda fn, key: lambda *a, **k: cur.__setitem__(key, fn(*a, **k)) or cur[key])(fn, key))
    pol, done, obs = MaxExpectedCapacityPolicy(0.0), False, None
    while not done:
        obs, _, done, _, _ = env.step(pol(env.core, obs))
        years.append({**cur, "load_scale": env.core.load_scale})
    return years


def build_milp(feeder: str, years: list[dict]):
    """Wait-and-see MILP for one scenario (maximisation written as minimisation).

    Variables: a[t,i] installations, x[t,i] cumulative capacity (<= potential),
    X[t] total capacity, z[t,d] daily net peaks, one binary per liftable bus-hour.
    """
    from scipy.sparse import coo_matrix
    from pvplan.network import Network33, V_MIN_PU
    from pvplan.profiles import bus_shapes, rho_potential
    cfg = EnvConfig(impedance_scale=IMP[feeder])
    net = Network33(impedance_scale=IMP[feeder])
    shapes = bus_shapes()
    N, T, B = 32, len(years), cfg.annual_budget_mw
    pot = rho_potential() * net.p_load_kw / 1000.0 * cfg.potential_scale
    D = years[0]["days"].shape[0]
    BH = N * D * 24
    ia = lambda t, i: t * N + i
    ix = lambda t, i: N * T + t * N + i
    iX = lambda t: 2 * N * T + t
    iz = lambda t, d: 2 * N * T + T + t * D + d
    nb0 = 2 * N * T + T + T * D
    rows, cols, vals, lb, ub = [], [], [], [], []
    nrow = 0

    def add(r_cols, r_vals, lo, hi):
        nonlocal nrow
        rows.extend([nrow] * len(r_cols)); cols.extend(r_cols); vals.extend(r_vals)
        lb.append(lo); ub.append(hi); nrow += 1

    cmap, const, trb, bins = {}, 0.0, 0.0, []
    for t, y in enumerate(years):
        for i in range(N):                       # x[t,i] = x[t-1,i] + a[t,i]
            add([ix(t, i), ia(t, i)] + ([ix(t - 1, i)] if t else []),
                [1.0, -1.0] + ([-1.0] if t else []), 0.0, 0.0)
        add([iX(t)] + [ix(t, i) for i in range(N)], [1.0] + [-1.0] * N, 0.0, 0.0)
        add([ia(tt, i) for tt in range(t + 1) for i in range(N)],          # cumulative budget
            [1.0 / max(years[tt]["xi"][i], 1e-3) for tt in range(t + 1) for i in range(N)],
            -np.inf, (t + 1) * B)
        gross = net.p_load_kw[:, None, None] * shapes[:, None, :] * y["load_scale"] * y["noise"]
        q = net.q_load_kvar[:, None, None] * shapes[:, None, :] * y["load_scale"] * y["noise"]
        sysg = gross.sum(axis=0)
        pg = sysg.max(axis=1).mean()
        days = y["days"]
        const += cfg.w_peak
        for d in range(D):                        # z[t,d] >= sysg[d,h] - 1000 X_t pv[d,h]
            cmap[iz(t, d)] = cfg.w_peak / (pg * D)        # minimise mean net peak
            for h in range(24):
                add([iz(t, d), iX(t)], [1.0, 1000.0 * days[d, h]], sysg[d, h], np.inf)
        vg = net.voltages_pu(gross.reshape(N, -1), q.reshape(N, -1)).reshape(N, D, 24)
        lift_max = (net.r_common_pu @ pot)[:, None, None] * days[None, :, :]
        need = V_MIN_PU - vg
        cand = (need > 0) & (days[None, :, :] > 0) & (lift_max >= need - 1e-12)
        bins += [(t, n, d, h, need[n, d, h]) for n, d, h in zip(*np.nonzero(cand))]
        load = np.hypot(gross, q) / net.tr_capacity_kva[:, None, None]
        trb += cfg.w_tr * float(((load > cfg.overload_threshold) & (days[None] > 0)).sum()) / BH
    for k, (t, n, d, h, nd) in enumerate(bins):  # pv * Rc[n] @ x_t >= need * b
        pv = years[t]["days"][d, h]
        add([nb0 + k] + [ix(t, i) for i in range(N)],
            [-nd] + list(pv * net.r_common_pu[n]), 0.0, np.inf)
        cmap[nb0 + k] = -cfg.w_v / BH
    nvar = nb0 + len(bins)
    cvec = np.zeros(nvar)
    for k, v in cmap.items():
        cvec[k] = v
    A = coo_matrix((vals, (rows, cols)), shape=(nrow, nvar)).tocsr()
    integ = np.r_[np.zeros(nb0), np.ones(len(bins))]
    lo = np.r_[np.zeros(2 * N * T + T), np.full(T * D, -np.inf), np.zeros(len(bins))]
    hi = np.r_[np.full(N * T, np.inf), np.tile(pot, T), np.full(T, np.inf),
               np.full(T * D, np.inf), np.ones(len(bins))]
    return cvec, A, np.array(lb), np.array(ub), integ, lo, hi, const + trb, trb, len(bins)


def replay(feeder: str, scenario: int, a: np.ndarray, years: list[dict]) -> float:
    """Return of the wait-and-see installations replayed in the environment."""
    from pvplan import pvgis
    from pvplan.environment import PlanningCore
    core = PlanningCore(pvgis.load_summer_days(), EnvConfig(impedance_scale=IMP[feeder]), scenario)
    total = 0.0
    for t in range(len(years)):
        offers = a[t] / np.maximum(years[t]["xi"], 1e-3)
        budget = core.cfg.annual_budget_mw + core.carryover_mw
        reserve = max(budget - offers.sum(), 0.0)
        total += core.apply_year(offers, reserve)["reward"]
    return total


def _bound_job(job: tuple) -> dict:
    import time
    from scipy.optimize import Bounds, LinearConstraint, milp
    feeder, scenario, tlim = job
    years = record_draws(feeder, scenario)
    cvec, A, lb, ub, integ, lo, hi, const, trb, nbin = build_milp(feeder, years)
    t0 = time.perf_counter()
    lp = milp(cvec, constraints=LinearConstraint(A, lb, ub), integrality=np.zeros_like(integ),
              bounds=Bounds(lo, hi))
    res = milp(cvec, constraints=LinearConstraint(A, lb, ub), integrality=integ,
               bounds=Bounds(lo, hi), options={"time_limit": tlim, "mip_rel_gap": 1e-4})
    out = {"feeder": feeder, "scenario_seed": scenario, "n_binaries": nbin,
           "lp_bound": const - lp.fun, "tr_bound_term": trb,
           "seconds": time.perf_counter() - t0, "status": res.status}
    if res.x is not None:
        a = res.x[:32 * len(years)].reshape(len(years), 32)
        out.update({"ws_value": const - res.fun,
                    "ws_bound": const - (res.fun if getattr(res, "mip_dual_bound", None) is None
                                             else res.mip_dual_bound),
                    "ws_replay": replay(feeder, scenario, a, years),
                    "ws_installed_mw": float(a.sum())})
    print(f"  {feeder} {scenario}: bins={nbin} LP={out['lp_bound']:.4f} "
          f"WS={out.get('ws_value', np.nan):.4f} bound={out.get('ws_bound', np.nan):.4f} "
          f"replay={out.get('ws_replay', np.nan):.4f} {out['seconds']:.0f}s", flush=True)
    return out


def cmd_bound(feeder: str, tlim: float, workers: int, n: int) -> None:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_bound_job, [(feeder, s, tlim) for s in SEEDS[:n]]))
    pd.DataFrame(rows).to_csv(RESULTS / f"bound_{feeder}.csv", index=False)


# ------------------------------------------------------------------ stats
def cmd_stats() -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired
    T = 10
    cfg = EnvConfig()

    def fam(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        d["family"] = d.policy.str.replace(r"-s\d$", "", regex=True)
        d["peak_term"] = cfg.w_peak * T * d.dpeak_frac_mean
        d["v_term"] = -cfg.w_v * d.volt_delta_sum
        return d.groupby(["family", "scenario_seed"])[["return_sum", "peak_term", "v_term",
                                                      "installed_total_mw"]].mean()

    ro = pd.read_csv(RESULTS / "rollout_evaluation.csv")
    oe = pd.concat([pd.read_csv(RESULTS / "oltc_evaluation.csv"),
                    pd.read_csv(RESULTS / "oltc_retrained_evaluation.csv")], ignore_index=True)
    oe["setting"] = oe.setting.astype(str)
    names = {"weak": ("none", {"rule-V": "max-exp-cap-V retuned", "rule": "max-exp-cap",
                               "residual": "residual", "ppo": "ppo", "lp-v": "rolling-lp-V"}),
             "weak-oltc102": ("1.02", {"rule-V": "max-exp-cap-V retuned", "rule": "max-exp-cap",
                                       "residual": "residual-rt", "ppo": "ppo-rt",
                                       "lp-v": "rolling-lp-V"})}
    rows, summary = [], {}
    for setting, (okey, m) in names.items():
        g = fam(oe[oe.setting == okey])
        pol = {k: g.loc[v] for k, v in m.items()}
        pol["rollout"] = fam(ro[ro.setting == setting]).loc["rollout"]
        for a, b in [("rollout", "rule-V"), ("residual", "rollout"), ("rollout", "rule"),
                     ("rollout", "ppo"), ("rollout", "lp-v"), ("residual", "rule-V")]:
            x, y = pol[a], pol[b].loc[pol[a].index]
            d = (x.return_sum - y.return_sum).values
            lo, hi = bootstrap_ci(d)
            rows.append({"setting": setting, "policy": a, "reference": b, "diff": d.mean(),
                         "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": float(sps.wilcoxon(x.return_sum, y.return_sum).pvalue),
                         "wins": int((d > 0).sum()),
                         "rrb": rank_biserial_paired(x.return_sum.values, y.return_sum.values),
                         "diff_peak_term": float((x.peak_term - y.peak_term).mean()),
                         "diff_v_term": float((x.v_term - y.v_term).mean())})
        paths = ro[ro.setting == setting].kappa_path.str.split(",", expand=True).astype(float)
        summary[setting] = {"means": {k: float(v.return_sum.mean()) for k, v in pol.items()},
                            "kappa_by_year": [float(v) for v in paths.mean(axis=0)],
                            "kappa_median_by_year": [float(v) for v in paths.median(axis=0)]}
    for feeder in ("stiff", "weak"):
        path = RESULTS / f"bound_{feeder}.csv"
        if not path.exists():
            continue
        b = pd.read_csv(path)
        if feeder == "stiff":
            nom = pd.concat([pd.read_csv(RESULTS / "evaluation_nominal.csv"),
                             pd.read_csv(RESULTS / "review_r1_nominal.csv")], ignore_index=True)
            g = fam(nom)
            means = {k: float(g.loc[k].return_sum.mean()) for k in ("max-exp-cap", "ppo", "oracle", "rolling-lp")}
        else:
            means = summary["weak"]["means"]
        summary[f"bound_{feeder}"] = {
            # ws_value: the best wait-and-see installations replayed in the simulator
            "n": int(len(b)), "ws_value": float(b.ws_replay.mean()), "ws_bound": float(b.ws_bound.mean()),
            "lp_bound": float(b.lp_bound.mean()), "replay_err_max": float((b.ws_replay - b.ws_value).abs().max()),
            "gap_rel_mean": float(((b.ws_bound - b.ws_value) / b.ws_bound).mean()),     # solver gap
            "ws_installed_mw": float(b.ws_installed_mw.mean()), "seconds_mean": float(b.seconds.mean()),
            "n_binaries_mean": float(b.n_binaries.mean()),
            "share_of_ws": {k: v / float(b.ws_replay.mean()) for k, v in means.items()},
            "share_of_bound": {k: v / float(b.ws_bound.mean()) for k, v in means.items()}}
        base = "max-exp-cap" if feeder == "stiff" else "rule"
        summary[f"bound_{feeder}"]["gap_closed"] = {
            k: (v - means[base]) / (float(b.ws_replay.mean()) - means[base]) for k, v in means.items()}
    sens_path = RESULTS / "rollout_sensitivity.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        ref = fam(oe[oe.setting == "none"]).loc["max-exp-cap-V retuned"].return_sum
        base_ro = fam(ro[ro.setting == "weak"]).loc["rollout"].return_sum
        summary["rollout_sensitivity"] = {}
        for v, g in sens.groupby("variant"):
            x = g.set_index("scenario_seed").return_sum.sort_index()
            out = {"mean": float(x.mean())}
            for name, y in (("vs_rule", ref), ("vs_base_rollout", base_ro)):
                d = (x - y.loc[x.index]).values
                lo, hi = bootstrap_ci(d)
                out[name] = {"diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi}
            summary["rollout_sensitivity"][v] = out
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "lookahead_stats.csv", index=False)
    (RESULTS / "lookahead_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(st.round(4).to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rollout", "rollout-sens", "bound", "stats"])
    ap.add_argument("--feeder", choices=["stiff", "weak"], default="weak")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "rollout":
        cmd_rollout(a.workers)
    elif a.cmd == "rollout-sens":
        cmd_rollout_sens(a.workers)
    elif a.cmd == "bound":
        cmd_bound(a.feeder, a.time_limit, a.workers, a.scenarios)
    else:
        cmd_stats()
