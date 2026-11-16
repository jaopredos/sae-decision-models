"""TopK sparse autoencoder (Gao et al., arXiv:2406.04093) for Laya's layer activations.

The SAE works in a rescaled space: inputs are multiplied by `scale` so that the mean input norm is
sqrt(d_in). `scale` is set once from data (`fit_scale`) and stored as a buffer, so a checkpoint is
self-contained. Public methods take and return activations in the model's own units; only the
latents live in the rescaled space (a latent value of 1.0 means "1.0 in rescaled units").

Training pieces follow the paper: decoder rows with unit norm, the gradient component parallel to
each row removed before the step, encoder initialised to the decoder transposed, `b_dec` at the
geometric median of a first batch, and an AuxK loss that lets the `k_aux` strongest *dead* latents
reconstruct the main reconstruction error, so that dead latents come back.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


def geometric_median(x: torch.Tensor, iters: int = 30, eps: float = 1e-6) -> torch.Tensor:
    """Weiszfeld iterations; x [n, d]."""
    m = x.mean(0)
    for _ in range(iters):
        w = 1.0 / (x - m).norm(dim=1).clamp_min(eps)
        m_new = (x * w[:, None]).sum(0) / w.sum()
        if (m_new - m).norm() < 1e-7 * m.norm().clamp_min(1.0):
            m = m_new
            break
        m = m_new
    return m


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_latents: int, k: int = 32, k_aux: int = 512, aux_coef: float = 1 / 32,
                 seed: int = 0):
        super().__init__()
        assert 0 < k <= n_latents
        self.d_in, self.n_latents, self.k = d_in, n_latents, k
        self.k_aux, self.aux_coef = k_aux, aux_coef
        g = torch.Generator().manual_seed(seed)
        dec = torch.randn(n_latents, d_in, generator=g)
        dec = dec / dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(dec)
        self.W_enc = nn.Parameter(dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("scale", torch.ones(()))
        # tokens whose activation norm exceeds this (model units) are "massive" (attention sinks such as
        # [CLS]) and are outside the SAE's domain: not trained on, not reconstructed, left alone by
        # `sae_intervention`. inf = no exclusion (the literal all-token protocol).
        self.register_buffer("massive_threshold", torch.tensor(float("inf")))
        # tokens since each latent last fired; used for the dead-latent mask, saved with the checkpoint
        self.register_buffer("tokens_since_fired", torch.zeros(n_latents))

    # ---- scaling -------------------------------------------------------------------------
    @torch.no_grad()
    def fit_scale(self, x: torch.Tensor) -> float:
        """Rescale so the mean input norm is sqrt(d_in)."""
        self.scale.fill_(math.sqrt(self.d_in) / x.float().norm(dim=1).mean().item())
        return self.scale.item()

    @torch.no_grad()
    def init_b_dec(self, x: torch.Tensor) -> None:
        self.b_dec.copy_(geometric_median((x.float() * self.scale)))

    def is_massive(self, x: torch.Tensor) -> torch.Tensor:
        """Bool [..]: token activation norm above `massive_threshold` (False everywhere if it is inf)."""
        return x.float().norm(dim=-1) > self.massive_threshold

    # ---- forward pieces (rescaled space) -------------------------------------------------
    def pre_acts(self, xs: torch.Tensor) -> torch.Tensor:
        return (xs - self.b_dec) @ self.W_enc + self.b_enc

    def topk(self, pre: torch.Tensor) -> torch.Tensor:
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))

    def decode_rescaled(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    # ---- public API, model units ---------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse latents [.., n_latents] (exact zeros off the top-k)."""
        return self.topk(self.pre_acts(x.float() * self.scale))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_rescaled(z) / self.scale

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ---- training ------------------------------------------------------------------------
    def training_loss(self, x: torch.Tensor, dead_window_tokens: float) -> Dict[str, torch.Tensor]:
        """x [n, d_in] in model units. Returns main/aux losses and bookkeeping tensors."""
        xs = x.float() * self.scale
        pre = self.pre_acts(xs)
        z = self.topk(pre)
        recon = self.decode_rescaled(z)
        err = xs - recon
        main = err.pow(2).mean()

        with torch.no_grad():
            fired = (z > 0).any(0)
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired[fired] = 0
        dead = self.tokens_since_fired > dead_window_tokens
        n_dead = int(dead.sum())
        aux = main.new_zeros(())
        if n_dead > 0:
            k_aux = min(self.k_aux, n_dead)
            pre_dead = pre.masked_fill(~dead, -torch.inf)
            vals, idx = pre_dead.topk(k_aux, dim=-1)
            z_aux = torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))
            err_hat = z_aux @ self.W_dec
            e = err.detach()
            aux = (e - err_hat).pow(2).mean() / e.pow(2).mean().clamp_min(1e-8)
        return {"loss": main + self.aux_coef * aux, "main": main.detach(), "aux": aux.detach(),
                "n_dead": torch.tensor(n_dead), "z": z.detach()}

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Project the decoder gradient onto the tangent space of the unit sphere (per row)."""
        if self.W_dec.grad is None:
            return
        g, w = self.W_dec.grad, self.W_dec
        g -= (g * w).sum(1, keepdim=True) * w

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

    # ---- evaluation ----------------------------------------------------------------------
    @torch.no_grad()
    def fve_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """Sums needed to pool FVE over batches: sse, sum, sumsq and n (model units)."""
        x = x.float()
        r = self.reconstruct(x)
        return {"sse": (x - r).pow(2).sum().item(), "sum": x.sum(0).cpu().double(),
                "sumsq": x.pow(2).sum().item(), "n": x.shape[0], "l0": (self.encode(x) > 0).sum(-1).float().mean().item()}


def pooled_fve(parts) -> float:
    """FVE = 1 - SSE / total variance, from `fve_stats` dicts of several batches."""
    n = sum(p["n"] for p in parts)
    sse = sum(p["sse"] for p in parts)
    s = sum(p["sum"] for p in parts)
    ss = sum(p["sumsq"] for p in parts)
    var = ss - float((s * s).sum()) / n
    return 1.0 - sse / var


def save_sae(sae: TopKSAE, path: str, extra: Optional[dict] = None) -> None:
    torch.save({"state_dict": sae.state_dict(),
                "config": {"d_in": sae.d_in, "n_latents": sae.n_latents, "k": sae.k, "k_aux": sae.k_aux,
                           "aux_coef": sae.aux_coef},
                "extra": extra or {}}, path)


def load_sae(path: str, device: str = "cpu") -> TopKSAE:
    ck = torch.load(path, map_location=device, weights_only=False)
    sae = TopKSAE(**ck["config"])
    sae.load_state_dict(ck["state_dict"])
    return sae.to(device)
