"""Hooks and subspace estimators for the position-sensitivity experiments (PROTOCOLO.md, emenda 7).

  between_class_directions  top directions of the between-class scatter of vectors X within groups (class = slot or option)
  random_subspace           random orthonormal subspace of the same dimension (null)
  ProjectOut                h' = h - (h U) U^T at selected positions of one layer output
  CaptureMarkers            records one layer output at the marker positions, call after call
  ReplaceMarkers            writes given vectors into the marker positions of one layer output
  ZeroHeads                 zeroes attention heads of one attention module (pre-hook on its output projection)

Like `src/sae_intervention.py`, a hooked layer's output may be a tensor or a tuple whose first element is the hidden state;
the arithmetic is done in float32 and cast back.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch


def between_class_directions(X: np.ndarray, grp: np.ndarray, cls: np.ndarray, energy: float = 0.9, r_max: int = 16):
    """Orthonormal U [d, r] spanning the pooled between-class offsets (class mean minus group mean, weighted by sqrt(n)).
    r is the smallest number of directions holding >= `energy` of the offsets' variance, at least 1 and at most r_max.
    Returns (U, r, explained) where explained[i] is the cumulative variance fraction of the first i+1 directions."""
    rows = []
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        x = X[idx].astype(np.float64)
        mu = x.mean(0)
        for c in np.unique(cls[idx]):
            m = cls[idx] == c
            rows.append(np.sqrt(m.sum()) * (x[m].mean(0) - mu))
    _, s, vt = np.linalg.svd(np.stack(rows), full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    r = int(min(max(1, np.searchsorted(cum, energy) + 1), r_max, len(s)))
    return vt[:r].T.astype(np.float32), r, cum[:r].astype(np.float32)


def random_subspace(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    return q.astype(np.float32)


def _split(output):
    is_tuple = isinstance(output, tuple)
    return (output[0] if is_tuple else output), is_tuple


def _join(h, output, is_tuple):
    return (h,) + tuple(output[1:]) if is_tuple else h


class _Hooked:
    def __init__(self, module):
        self.module, self._handle = module, None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None


class ProjectOut(_Hooked):
    """Remove the span of U from the hidden state at the positions where `mask(h)` ([B, S] bool) holds (None = everywhere)."""

    def __init__(self, layer, U: np.ndarray, mask: Optional[Callable] = None):
        super().__init__(layer)
        self.U, self.mask = torch.as_tensor(U, dtype=torch.float32), mask

    def _hook(self, _m, _i, output):
        h, is_tuple = _split(output)
        h32, U = h.float(), self.U.to(h.device)
        new = h32 - (h32 @ U) @ U.T
        keep = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device) if self.mask is None else self.mask(h).to(h.device).bool()
        return _join(torch.where(keep[..., None], new, h32).to(h.dtype), output, is_tuple)


class CaptureMarkers(_Hooked):
    """Append, per call, the layer output at the marker positions: tensor [rows, K, H] (float32, CPU). `marker_pos` is a
    callable returning the [rows, K] long positions of the current call (-1 = no marker)."""

    def __init__(self, layer, marker_pos: Callable):
        super().__init__(layer)
        self.marker_pos, self.chunks = marker_pos, []

    def _hook(self, _m, _i, output):
        h, _ = _split(output)
        pos = self.marker_pos().to(h.device)
        idx = pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        self.chunks.append(torch.gather(h.float(), 1, idx).cpu())

    def result(self) -> torch.Tensor:
        return torch.cat(self.chunks)


class ReplaceMarkers(_Hooked):
    """Write vectors into the marker positions. `source()` returns (pos [rows, K] long, vec [rows, K, H], valid [rows, K] bool)
    for the current call; only valid entries are replaced."""

    def __init__(self, layer, source: Callable):
        super().__init__(layer)
        self.source = source

    def _hook(self, _m, _i, output):
        h, is_tuple = _split(output)
        pos, vec, valid = (t.to(h.device) for t in self.source())
        h32 = h.float().clone()
        b, j = valid.nonzero(as_tuple=True)
        h32[b, pos[b, j]] = vec[b, j].to(h32)
        return _join(h32.to(h.dtype), output, is_tuple)


class ZeroHeads:
    """Zero the output of heads `heads` of an attention module whose output projection is `attn.Wo`."""

    def __init__(self, attn, heads, n_heads: int):
        self.attn, self.heads, self.n_heads, self._handle = attn, list(heads), n_heads, None

    def _pre(self, _m, args):
        x = args[0]
        y = x.clone().view(*x.shape[:-1], self.n_heads, x.shape[-1] // self.n_heads)
        y[..., self.heads, :] = 0
        return (y.view_as(x),) + tuple(args[1:])

    def __enter__(self):
        self._handle = self.attn.Wo.register_forward_pre_hook(self._pre)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None
