"""Linear option scorer probes over Laya's per-layer marker activations (stage B).

A probe scores every option marker of a decision with `s = w·h + b` and takes a softmax over the
options of that decision, trained with cross-entropy against the soft gold distribution plus
`wd·||w||²`. The number of options varies (2, 4 or 5), so decisions are padded to the widest one
and masked. `b` is kept to match the w·h + b spec, but it adds the same amount to every option of
a decision and so cancels in the softmax: its gradient is exactly zero.

The objective is convex (strictly, for wd > 0), so full-batch L-BFGS reaches the same optimum from
any init; the seed only moves the init, and the shuffled-label permutation of the control.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

NEG = -1e9  # masked option logit; finite so that 0 * log_softmax stays 0, not nan
EPS = 1e-9


@dataclass
class SplitData:
    dec: pd.DataFrame          # one row per decision, ordered by decision_idx
    marker_rows: np.ndarray    # [n_dec, max_k] row into the layer .npy, -1 where padded
    mask: np.ndarray           # [n_dec, max_k] bool
    gold: np.ndarray           # [n_dec, max_k] float, 0 where padded
    gold_idx: np.ndarray       # [n_dec] index of gold_argmax among option_ids
    layers: List[str]          # file stems, L{idx}_{label}
    root: Path

    def layer(self, stem: str) -> np.ndarray:
        """[n_dec, max_k, hidden] float32, zeros where padded."""
        x = np.load(self.root / f"{stem}.npy").astype(np.float32)
        out = np.zeros(self.marker_rows.shape + (x.shape[1],), dtype=np.float32)
        out[self.mask] = x[self.marker_rows[self.mask]]
        return out


def load_split(root: Path) -> SplitData:
    dec = pd.read_parquet(root / "decisions.parquet").sort_values("decision_idx").reset_index(drop=True)
    assert (dec["decision_idx"].to_numpy() == np.arange(len(dec))).all()
    mk = pd.read_parquet(root / "markers.parquet")
    max_k = int(dec["n_options"].max())
    rows = np.full((len(dec), max_k), -1, dtype=np.int64)
    rows[mk["decision_idx"].to_numpy(), mk["option_index"].to_numpy()] = np.arange(len(mk))
    mask = rows >= 0
    assert (mask.sum(1) == dec["n_options"].to_numpy()).all()
    gold = np.zeros((len(dec), max_k), dtype=np.float32)
    gold_idx = np.zeros(len(dec), dtype=np.int64)
    for i, (opts, dist, lab) in enumerate(zip(dec["option_ids"], dec["gold_dist"], dec["gold_argmax"])):
        gold[i, : len(dist)] = dist
        gold_idx[i] = list(opts).index(lab)
    gold /= gold.sum(1, keepdims=True)
    layers = sorted(p.stem for p in root.glob("L[0-9][0-9]_*.npy"))
    return SplitData(dec, rows, mask, gold, gold_idx, layers, root)


def fit_probe(x: torch.Tensor, mask: torch.Tensor, gold: torch.Tensor, wd: float, seed: int,
              max_iter: int = 500) -> Dict[str, torch.Tensor]:
    """x [n, k, h] (already standardized), mask [n, k], gold [n, k]. Returns {"w", "b"}."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(x.shape[-1], generator=g) * 0.01).to(x.device).requires_grad_(True)
    b = torch.zeros(1, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logp = torch.log_softmax((x @ w + b).masked_fill(~mask, NEG), -1)
        loss = -(gold * logp).sum(-1).mean() + wd * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"w": w.detach(), "b": b.detach()}


@torch.no_grad()
def predict_probe(params, x: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    s = (x @ params["w"] + params["b"]).masked_fill(~mask, NEG)
    return torch.softmax(s, -1).cpu().numpy()


def standardizer(x: np.ndarray, mask: np.ndarray):
    """Mean/std over the real (unpadded) markers of the fitting set; returns apply(arr, mask)."""
    v = x[mask]
    mu, sd = v.mean(0), v.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return lambda arr, m: np.where(m[..., None], (arr - mu) / sd, 0.0).astype(np.float32)


def metrics(probs: np.ndarray, gold: np.ndarray, gold_idx: np.ndarray, mask: np.ndarray,
            dec: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """acc vs gold_argmax, soft acc = mean gold[pred], KL(gold || pred); optional breakdowns."""
    p = np.where(mask, probs, 0.0)
    pred = p.argmax(1)
    correct = pred == gold_idx
    soft = gold[np.arange(len(gold)), pred]
    kl = np.where(gold > 0, gold * (np.log(np.clip(gold, EPS, None)) - np.log(np.clip(p, EPS, None))), 0.0).sum(1)
    out = {"acc": float(correct.mean()), "soft_acc": float(soft.mean()), "kl": float(kl.mean())}
    if dec is not None:
        for col in ("qtype", "workflow"):
            for key, idx in dec.groupby(col).indices.items():
                out[f"acc_{key}"] = float(correct[idx].mean())
    return out


def shuffle_within_question(gold: np.ndarray, dec: pd.DataFrame, seed: int) -> np.ndarray:
    """Control labels: gold distributions permuted among the decisions of the same
    (workflow, question_id), so option count and per-question marginals are preserved."""
    rng = np.random.default_rng(seed)
    out = gold.copy()
    for _, idx in dec.groupby(["workflow", "question_id"]).indices.items():
        out[idx] = gold[rng.permutation(idx)]
    return out


def prior_probs(train: SplitData, target: SplitData, pseudo: float = 0.5) -> np.ndarray:
    """Per (workflow, question_id): frequency of the gold label in train, add-`pseudo` smoothed."""
    probs = np.zeros(target.gold.shape, dtype=np.float64)
    counts = {}
    for (wf, qid), idx in train.dec.groupby(["workflow", "question_id"]).indices.items():
        k = int(train.mask[idx[0]].sum())
        c = np.bincount(train.gold_idx[idx], minlength=k).astype(np.float64) + pseudo
        counts[(wf, qid)] = c / c.sum()
    for i, (wf, qid) in enumerate(zip(target.dec["workflow"], target.dec["question_id"])):
        c = counts[(wf, qid)]
        probs[i, : len(c)] = c
    return probs


def uniform_probs(target: SplitData) -> np.ndarray:
    return target.mask / target.mask.sum(1, keepdims=True)


def padded(values, shape) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float64)
    for i, v in enumerate(values):
        out[i, : len(v)] = v
    return out
