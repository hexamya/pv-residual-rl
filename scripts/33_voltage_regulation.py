"""The weak feeder with a substation OLTC (line-drop compensation).

The unregulated weak feeder holds the feeder head at 1.0 pu. Here an on-load
tap changer (0.625% steps) regulates the estimated load-centre voltage to
1.0 pu every hour, with the head voltage capped at v_max (the regulation
margin left after the low-voltage network and light-load limits). v_max = 1.0
is the unregulated case; v_max = 1.05 uses the whole +/-5% band.

screen: re-tunes the rule's kappa per setting on the tuning scenarios and
        evaluates the rules, the rolling LP-V (nominal lambda), and the
        residual and plain PPO agents trained on the unregulated feeder (frozen)
        on the 100 test scenarios.
train:  retrains the residual agent (on the re-tuned rule) and plain PPO in the
        settings of RETRAIN, three seeds each.
eval:   evaluates the retrained agents and writes the paired statistics.

Usage:
  python scripts/33_voltage_regulation.py screen [--workers 4]
  python scripts/33_voltage_regulation.py train --cap 1.02 --kind residual --seed 0
  python scripts/33_voltage_regulation.py eval [--workers 4]
  python scripts/33_voltage_regulation.py stats      # recompute statistics only
Outputs: results/oltc_tuning.csv, oltc_kappa.json, oltc_evaluation.csv,
         oltc_retrained_evaluation.csv, oltc_stats.csv
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pvplan.environment import EnvConfig

RESULTS = ROOT / "results"
SETTINGS = ["none", "1.01", "1.02", "1.03", "1.04", "1.05"]
RETRAIN = ["1.02"]
N_RT = 3
BUS_HOURS_PER_SUMMER = 32 * 24 * 92
T = 10


def cfg(setting: str) -> EnvConfig:
    return EnvConfig(impedance_scale=1.0,
                     oltc_vmax=None if setting == "none" else float(setting))


def ctag(setting: str, kind: str) -> str:
    return f"oltc_{setting.replace('.', '')}_{kind}"


def _mods():
    return importlib.import_module("17_weak_feeder"), importlib.import_module("18_hybrid")


def _policy(setting: str, label: str):
    weak, hyb = _mods()
    nominal = json.loads((RESULTS / "weak_tuned.json").read_text())
    fam, _, seed = label.rpartition("-s")
    if label == "max-exp-cap":
        return weak.make_policy("max-exp-cap", 0.0)
    if label.startswith("max-exp-cap-V k="):
        return weak.make_policy("max-exp-cap", float(label.split("=")[1]))
    if label == "max-exp-cap-V":
        return weak.make_policy("max-exp-cap", nominal["kappa"])
    if label == "max-exp-cap-V retuned":
        kappa = json.loads((RESULTS / "oltc_kappa.json").read_text())[setting]
        return weak.make_policy("max-exp-cap", kappa)
    if label == "rolling-lp-V":
        return weak.make_policy("rolling-lp-v", nominal["lam_v"])
    if fam == "residual":
        return hyb.HybridPolicy("residual", int(seed))
    if fam == "ppo":
        return weak.make_policy(label)
    if fam == "residual-rt":
        kappa = json.loads((RESULTS / "oltc_kappa.json").read_text())[setting]
        return hyb.HybridPolicy("residual", int(seed), tag=ctag(setting, "residual"), kappa=kappa)
    if fam == "ppo-rt":
        return weak.SB3Policy(RESULTS / "logs" / ctag(setting, "ppo") / f"ppo_seed{seed}"
                              / "best_model.zip")
    raise ValueError(label)


def _job(job: tuple) -> pd.DataFrame:
    import torch
    from pvplan import pvgis
    from pvplan.evaluation import evaluate_policy
    torch.set_num_threads(1)
    setting, label, seeds = job
    df = evaluate_policy(label, _policy(setting, label), pvgis.load_summer_days(),
                         cfg(setting), seeds)
    df["setting"] = setting
    return df


def run(jobs: list, workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.concat(list(ex.map(_job, jobs)), ignore_index=True)


def cmd_screen(workers: int) -> None:
    weak, _ = _mods()
    tune = run([(s, f"max-exp-cap-V k={k}", weak.SEEDS_TUNE)
                for s in SETTINGS for k in weak.KAPPAS], workers)
    tune["kappa"] = tune.policy.str.split("=").str[1].astype(float)
    tune.to_csv(RESULTS / "oltc_tuning.csv", index=False)
    means = tune.groupby(["setting", "kappa"])["return_sum"].mean().unstack("kappa")
    best = {s: float(means.loc[s].idxmax()) for s in SETTINGS}
    (RESULTS / "oltc_kappa.json").write_text(json.dumps(best, indent=2))
    print(means.round(4).to_string(), "\nselected kappa:", best, flush=True)

    seeds = weak.SEEDS_TEST
    labels = (["max-exp-cap", "max-exp-cap-V", "max-exp-cap-V retuned"]
              + [f"residual-s{k}" for k in range(5)] + [f"ppo-s{k}" for k in range(5)])
    jobs = [(s, lab, seeds) for s in SETTINGS for lab in labels]
    jobs += [(s, "rolling-lp-V", seeds[i:i + 25]) for s in SETTINGS for i in range(0, 100, 25)]
    df = run(jobs, workers)
    df.to_csv(RESULTS / "oltc_evaluation.csv", index=False)
    summarize(df)


def cmd_train(setting: str, kind: str, seed: int, steps: int) -> None:
    _, hyb = _mods()
    if kind == "residual":
        kappa = json.loads((RESULTS / "oltc_kappa.json").read_text())[setting]
        hyb.train_one("residual", seed, steps, cfg=cfg(setting), kappa=kappa,
                      tag=ctag(setting, "residual"))
    else:
        importlib.import_module("05_train_rl").train_one(
            "ppo", seed, steps, cfg=cfg(setting), tag=ctag(setting, "ppo"))


def cmd_eval(workers: int) -> None:
    weak, _ = _mods()
    jobs = [(s, f"{k}-s{i}", weak.SEEDS_TEST)
            for s in RETRAIN for k in ("residual-rt", "ppo-rt") for i in range(N_RT)]
    rt = run(jobs, workers)
    rt.to_csv(RESULTS / "oltc_retrained_evaluation.csv", index=False)
    summarize(pd.concat([pd.read_csv(RESULTS / "oltc_evaluation.csv"), rt], ignore_index=True))


def summarize(df: pd.DataFrame) -> None:
    from scipy import stats as sps
    from pvplan.stats import bootstrap_ci, rank_biserial_paired
    df = df.copy()
    df["setting"] = df.setting.astype(str)
    df["family"] = df.policy.str.replace(r"-s\d$", "", regex=True)
    df["uv_relieved_bh"] = -df.volt_delta_sum / T * BUS_HOURS_PER_SUMMER
    df["peak_term"] = df.dpeak_frac_mean * T * EnvConfig().w_peak
    df["v_term"] = -EnvConfig().w_v * df.volt_delta_sum
    fam = df.groupby(["setting", "family", "scenario_seed"])[
        ["return_sum", "uv_relieved_bh", "volt_gross_sum", "dpeak_mw_final",
         "peak_term", "v_term"]].mean()
    ref = "max-exp-cap-V retuned"
    pairs = [(ref, "max-exp-cap"), ("residual", ref), ("residual-rt", ref),
             ("ppo", ref), ("ppo-rt", ref), ("residual-rt", "ppo-rt"),
             ("rolling-lp-V", ref), ("residual-rt", "residual")]
    rows = []
    for s in SETTINGS:
        g = fam.loc[s]
        fams = set(g.index.get_level_values(0))
        uv = g.loc["max-exp-cap"]["volt_gross_sum"] / T
        for a, b in pairs:
            if a not in fams or b not in fams:
                continue
            x, y = g.loc[a], g.loc[b].loc[g.loc[a].index]
            d = (x.return_sum - y.return_sum).values
            lo, hi = bootstrap_ci(d)
            rows.append({
                "setting": s, "policy": a, "reference": b,
                "uv_share_nopv": float(uv.mean()),
                "uv_bh_nopv": float(uv.mean() * BUS_HOURS_PER_SUMMER),
                "ref_return": float(y.return_sum.mean()),
                "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                "wilcoxon_p": 1.0 if np.allclose(d, 0)
                else float(sps.wilcoxon(x.return_sum.values, y.return_sum.values).pvalue),
                "wins": int((d > 0).sum()),
                "rrb": rank_biserial_paired(x.return_sum.values, y.return_sum.values),
                "diff_uv_bh": float((x.uv_relieved_bh - y.uv_relieved_bh).mean()),
                "diff_peak_kw": float(1000 * (x.dpeak_mw_final - y.dpeak_mw_final).mean()),
                "pol_peak_term": float(x.peak_term.mean()), "ref_peak_term": float(y.peak_term.mean()),
                "diff_peak_term": float((x.peak_term - y.peak_term).mean()),
                "diff_v_term": float((x.v_term - y.v_term).mean()),
            })
    st = pd.DataFrame(rows)
    st.to_csv(RESULTS / "oltc_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(st.drop(columns=["rrb", "wilcoxon_p"]).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "train", "eval", "stats"])
    ap.add_argument("--cap", choices=SETTINGS)
    ap.add_argument("--kind", choices=["residual", "ppo"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "screen":
        cmd_screen(a.workers)
    elif a.cmd == "train":
        cmd_train(a.cap, a.kind, a.seed, a.steps)
    elif a.cmd == "eval":
        cmd_eval(a.workers)
    else:
        summarize(pd.concat([pd.read_csv(RESULTS / "oltc_evaluation.csv"),
                             pd.read_csv(RESULTS / "oltc_retrained_evaluation.csv")],
                            ignore_index=True))
