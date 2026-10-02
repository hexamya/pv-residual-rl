"""Statistical comparison utilities (decision D13).

Paired design: every policy sees identical scenario seeds, so per-seed metric
differences are paired observations. We report Wilcoxon signed-rank p-values
(proposal 8-5-3), Cliff's delta effect sizes, and bootstrap CIs of the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]; >0 means x tends to exceed y."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1] (Kerby 2014).

    (R+ - R-) / (R+ + R-) over the ranks of |x - y|, zero differences dropped;
    the effect size that matches the Wilcoxon signed-rank test.
    """
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def paired_comparison_table(df: pd.DataFrame, metric: str,
                            reference: str) -> pd.DataFrame:
    """Compare every policy against `reference` on `metric`, paired by seed."""
    ref = (df[df.policy == reference]
           .set_index("scenario_seed")[metric].sort_index())
    rows = []
    for pol, g in df.groupby("policy"):
        if pol == reference:
            continue
        x = g.set_index("scenario_seed")[metric].sort_index()
        common = ref.index.intersection(x.index)
        a, b = x.loc[common].values, ref.loc[common].values
        diff = a - b
        if np.allclose(diff, 0):
            w_p = 1.0
        else:
            w_p = float(sps.wilcoxon(a, b).pvalue)
        lo, hi = bootstrap_ci(diff)
        rows.append({
            "policy": pol,
            "reference": reference,
            "metric": metric,
            "mean": float(a.mean()),
            "mean_ref": float(b.mean()),
            "mean_diff": float(diff.mean()),
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "wilcoxon_p": w_p,
            "cliffs_delta": cliffs_delta(a, b),
            "n_pairs": len(common),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False)
