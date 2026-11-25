"""Inject a TopK SAE into one layer of a model by a forward hook, and edit features there.

Used for the downstream-fidelity measurements of stage 4 and meant to be reused, unchanged, by the
causal ablation that comes after. Modes (all act only at the positions selected by `position_mask`):

  edit     h' = h + decode(z_edited) - decode(z). The SAE's reconstruction error stays in
           the stream, so with no edits h' is exactly h, and an edit moves the layer output only by
           the decoder directions of the edited features. This is the mode for ablations.
  replace  h' = decode(z) (the SAE reconstruction replaces the activation; error is dropped).
  mean     h' = a fixed vector (typically the dataset mean activation): a scale for "replace".
  zero     h' = 0, the other scale.

Edits are {feature_index: ("zero",) | ("clamp", value)}. A clamp value is in the SAE's rescaled
latent units (see src/sae.py), and also switches the feature on at positions where it was off.

Positions where the SAE's `is_massive` holds are left untouched unless `skip_massive=False`.
The hooked module's output may be a tensor or a tuple whose first element is the hidden states.
The arithmetic is done in float32 and cast back to the hook's input dtype.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.sae import TopKSAE

Edit = Tuple
MaskArg = Union[None, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


class SAEIntervention:
    def __init__(self, sae: TopKSAE, layer: nn.Module, mode: str = "edit",
                 edits: Optional[Dict[int, Edit]] = None, position_mask: MaskArg = None,
                 replacement: Optional[torch.Tensor] = None, skip_massive: bool = True):
        assert mode in ("edit", "replace", "mean", "zero"), mode
        if mode == "mean":
            assert replacement is not None, "mode='mean' needs `replacement`"
        self.sae, self.layer, self.mode = sae, layer, mode
        self.edits = dict(edits or {})
        self.position_mask = position_mask
        self.replacement = replacement
        self.skip_massive = skip_massive  # leave the SAE's "massive" tokens (see TopKSAE) untouched
        self._handle = None

    # ---- positions -----------------------------------------------------------------------
    def _mask(self, h: torch.Tensor) -> torch.Tensor:
        m = self.position_mask
        if m is None:
            return torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        m = m(h) if callable(m) else m
        return m.to(h.device).bool()

    def _apply_edits(self, z: torch.Tensor) -> torch.Tensor:
        z = z.clone()
        for feat, spec in self.edits.items():
            if spec[0] == "zero":
                z[..., feat] = 0.0
            elif spec[0] == "clamp":
                z[..., feat] = float(spec[1])
            else:
                raise ValueError(f"unknown edit {spec!r}")
        return z

    # ---- hook ----------------------------------------------------------------------------
    def _hook(self, _module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h32 = h.float()
        if self.mode == "zero":
            new = torch.zeros_like(h32)
        elif self.mode == "mean":
            new = self.replacement.to(h32).expand_as(h32)
        else:
            z = self.sae.encode(h32)
            if self.mode == "replace":
                new = self.sae.decode(z)
            elif not self.edits:
                new = h32
            else:
                new = h32 + (self.sae.decode(self._apply_edits(z)) - self.sae.decode(z))
        mask = self._mask(h)
        if self.skip_massive:
            mask = mask & ~self.sae.is_massive(h32)
        out = torch.where(mask[..., None], new, h32).to(h.dtype)
        return (out,) + tuple(output[1:]) if is_tuple else out

    def __enter__(self):
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None


@contextmanager
def intervene(sae: TopKSAE, layer: nn.Module, **kwargs):
    with SAEIntervention(sae, layer, **kwargs) as iv:
        yield iv
