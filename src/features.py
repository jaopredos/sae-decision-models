"""Per-feature statistics linking SAE features to Laya's errors (stage 5), SAEfarer-style.

A "view" gives, for every decision, the activation of every feature at some place: the marker of the
option Laya predicted, the marker of the correct option, or the maximum over the state tokens. A feature
is *active* in a decision when its activation is > 0 (TopK gives exact zeros). For a boolean outcome `y`
(here: Laya's answer is wrong) the table reports, per feature:

  binary    n_active, n_active_y, the outcome rate among the active decisions with its Wilson interval, the
            baseline rate and the lift, a hypergeometric (= Fisher exact) p-value for "more y than chance"
            and for "less", the two-sided p as twice the smaller one (capped at 1), and Benjamini-Hochberg q
            over the features with support >= `min_support`;
  weighted  the same rate with each decision weighted by the feature's activation, and the mean activation
            among y and among not-y decisions.

`confusion_enrichment` finds, per feature, the (question, predicted -> correct) pair that is most
over-represented among its active decisions. Everything runs on tensors, so 12k features x 5k decisions is
a few matrix products.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    """Wilson score interval of a proportion; n = 0 gives (nan, nan)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def hypergeom_p(k: np.ndarray, total: int, n_y: int, n_active: np.ndarray):
    """P(X >= k) and P(X <= k) of the number of y among `n_active` draws from `total` with `n_y` y.
    This is Fisher's exact one-sided test of "active" vs "outcome" on the 2x2 table."""
    rv = stats.hypergeom(M=total, n=n_y, N=n_active)
    return rv.sf(k - 1), rv.cdf(k)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (nan stays nan)."""
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
    return q


@torch.no_grad()
def enrichment_table(acts: torch.Tensor, y: torch.Tensor, min_support: int = 20) -> pd.DataFrame:
    """acts [N, F] activations (>= 0), y [N] bool. One line per feature, see the module docstring."""
    acts = acts.float()
    yf = y.float()
    active = (acts > 0).float()
    n = acts.shape[0]
    n_y = int(y.sum())
    n_act = active.sum(0).cpu().numpy()
    n_act_y = (active.T @ yf).cpu().numpy()
    sum_a = acts.sum(0).cpu().numpy()
    sum_a_y = (acts.T @ yf).cpu().numpy()
    base = n_y / n if n else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = n_act_y / n_act
        w_rate = sum_a_y / sum_a
        mean_y = sum_a_y / max(n_y, 1)
        mean_not_y = (sum_a - sum_a_y) / max(n - n_y, 1)
    lo, hi = wilson(n_act_y, n_act)
    p_hi, p_lo = hypergeom_p(n_act_y, n, n_y, n_act)
    supported = n_act >= min_support
    p_two = np.where(supported, np.minimum(1.0, 2 * np.minimum(p_hi, p_lo)), np.nan)
    return pd.DataFrame({
        "feature": np.arange(acts.shape[1]), "n_decisions": n, "n_y": n_y, "base_rate": base,
        "n_active": n_act.astype(int), "n_active_y": n_act_y.astype(int), "rate_active": rate,
        "wilson_lo": lo, "wilson_hi": hi, "lift": rate / base if base else np.nan,
        "p_more_y": np.where(supported, p_hi, np.nan), "p_less_y": np.where(supported, p_lo, np.nan),
        "p_two_sided": p_two, "q_bh": bh(p_two), "supported": supported,
        "weighted_rate": w_rate, "mean_act_y": mean_y, "mean_act_not_y": mean_not_y,
    })


@torch.no_grad()
def confusion_enrichment(acts: torch.Tensor, cls: torch.Tensor, n_classes: int, min_support: int = 20) -> pd.DataFrame:
    """Most over-represented confusion class among the decisions where each feature is active.

    cls [N] int: -1 for correct decisions, else the id of the (question, predicted, correct) class. Classes
    are tested against all decisions (hypergeometric, upper tail); the class with the smallest p is kept."""
    active = (acts.float() > 0).float()
    n = acts.shape[0]
    err = cls >= 0
    onehot = torch.zeros(n, n_classes, device=acts.device)
    onehot[err.nonzero().squeeze(1), cls[err]] = 1.0
    counts = (active.T @ onehot).cpu().numpy()            # [F, C]
    totals = onehot.sum(0).cpu().numpy()                  # [C]
    n_act = active.sum(0).cpu().numpy()                   # [F]
    p = np.ones_like(counts)
    for c in range(n_classes):
        if totals[c] == 0:
            continue
        p[:, c] = stats.hypergeom(M=n, n=int(totals[c]), N=n_act).sf(counts[:, c] - 1)
    best = p.argmin(1)
    rows = np.arange(len(best))
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (counts[rows, best] / n_act) / (totals[best] / n)
    return pd.DataFrame({"feature": rows, "top_confusion_class": best, "top_confusion_count": counts[rows, best].astype(int),
                         "top_confusion_lift": lift, "top_confusion_p": np.where(n_act >= min_support, p[rows, best], np.nan)})


def top_by(df: pd.DataFrame, column: str, k: int = 30, ascending: bool = False, require_supported: bool = True,
           q_max: Optional[float] = None) -> pd.DataFrame:
    d = df[df["supported"]] if require_supported else df
    if q_max is not None:
        d = d[d["q_bh"] <= q_max]
    return d.sort_values(column, ascending=ascending).head(k)


@torch.no_grad()
def cmh_table(acts: torch.Tensor, y: torch.Tensor, strata: torch.Tensor, min_support: int = 20) -> pd.DataFrame:
    """Cochran-Mantel-Haenszel test of "feature active" vs outcome `y`, controlling for `strata`.

    The plain enrichment in `enrichment_table` mixes a feature's link to the outcome with its link to whatever
    defines the strata (here: the question and the option Laya picked, which carry most of Laya's prior bias).
    CMH pools the 2x2 tables of the strata: z = sum(a_s - E_s) / sqrt(sum V_s) (a_s = active and y, E_s and V_s the
    hypergeometric mean and variance), two-sided normal p, and the Mantel-Haenszel common odds ratio. Strata of
    size 1 are ignored. `strata` [N] holds integer ids 0..S-1."""
    active = (acts.float() > 0).float()
    n_s = int(strata.max()) + 1
    onehot = torch.zeros(len(strata), n_s, device=acts.device)
    onehot[torch.arange(len(strata), device=acts.device), strata] = 1.0
    yf = y.float()
    N = onehot.sum(0)                                   # [S]
    m1 = onehot.T @ yf                                  # y in the stratum
    n1 = active.T @ onehot                              # [F, S] active in the stratum
    a = (active * yf[:, None]).T @ onehot               # [F, S] active and y
    ok = N > 1
    Ns = N.clamp_min(2)
    e = n1 * (m1 / Ns)
    v = n1 * (Ns - n1) * (m1 * (Ns - m1)) / (Ns ** 2 * (Ns - 1))
    e, v, a = e * ok, v * ok, a * ok
    b, c = n1 - a, m1[None, :] - a
    d = Ns[None, :] - n1 - m1[None, :] + a
    num = (a * d / Ns).sum(1); den = (b * c / Ns).sum(1)
    var = v.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = ((a - e).sum(1) / var.sqrt()).cpu().numpy()
        orr = (num / den).cpu().numpy()
    n_act = n1.sum(1).cpu().numpy()
    p = 2 * stats.norm.sf(np.abs(z))
    supported = (n_act >= min_support) & (var.cpu().numpy() > 0)
    p = np.where(supported, p, np.nan)
    return pd.DataFrame({"feature": np.arange(acts.shape[1]), "cmh_z": np.where(supported, z, np.nan),
                         "cmh_or": np.where(supported, orr, np.nan), "cmh_p": p, "cmh_q": bh(p)})
