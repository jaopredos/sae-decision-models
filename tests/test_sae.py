import torch

from src.sae import TopKSAE, pooled_fve, save_sae, load_sae


def _data(n=4096, d=16, seed=0):
    """Sparse-ish data: each point is a sum of 3 of 24 fixed directions, so a SAE can learn it."""
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(24, d, generator=g)
    basis /= basis.norm(dim=1, keepdim=True)
    idx = torch.stack([torch.randperm(24, generator=g)[:3] for _ in range(n)])
    return (basis[idx] * (1 + torch.rand(n, 3, 1, generator=g))).sum(1) * 5 + 2.0


def test_topk_has_exactly_k_active_and_nonnegative():
    sae = TopKSAE(16, 64, k=5)
    z = sae.encode(_data(128))
    assert ((z > 0).sum(-1) <= 5).all() and (z >= 0).all()
    assert ((z != 0).sum(-1) == 5).float().mean() > 0.5  # relu can zero a pre-act, never exceed k


def test_decoder_unit_norm_after_normalize_and_grad_is_tangent():
    sae = TopKSAE(16, 64, k=5)
    sae.W_dec.data *= torch.rand(64, 1) * 3 + 0.1
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-6)
    sae.W_dec.grad = torch.randn_like(sae.W_dec)
    sae.remove_parallel_grad()
    assert ((sae.W_dec.grad * sae.W_dec).sum(1).abs() < 1e-5).all()


def test_scale_gives_mean_norm_sqrt_d():
    x = _data()
    sae = TopKSAE(16, 64, k=5)
    sae.fit_scale(x)
    assert abs((x * sae.scale).norm(dim=1).mean().item() - 4.0) < 1e-4  # sqrt(16)


def test_training_raises_fve_and_keeps_decoder_normalised():
    torch.manual_seed(0)
    x = _data()
    sae = TopKSAE(16, 48, k=4, k_aux=8)
    sae.fit_scale(x)
    sae.init_b_dec(x)
    opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    fve0 = pooled_fve([sae.fve_stats(x)])
    for step in range(600):
        b = x[torch.randint(0, len(x), (512,))]
        out = sae.training_loss(b, dead_window_tokens=20_000)
        opt.zero_grad()
        out["loss"].backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
    fve1 = pooled_fve([sae.fve_stats(x)])
    assert fve1 > fve0 + 0.3 and fve1 > 0.6, (fve0, fve1)
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(48), atol=1e-5)


def test_auxk_revives_dead_latents():
    """With a tiny dead window every non-firing latent counts as dead and gets an aux gradient."""
    x = _data()
    sae = TopKSAE(16, 48, k=4, k_aux=8)
    sae.fit_scale(x)
    out = sae.training_loss(x[:256], dead_window_tokens=100)
    assert int(out["n_dead"]) > 0 and out["aux"] > 0


def test_pooled_fve_matches_direct():
    x = _data(300)
    sae = TopKSAE(16, 32, k=4)
    sae.fit_scale(x)
    parts = [sae.fve_stats(x[:100]), sae.fve_stats(x[100:])]
    r = sae.reconstruct(x)
    direct = 1 - (x - r).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
    assert abs(pooled_fve(parts) - direct.item()) < 1e-4


def test_save_load_roundtrip(tmp_path):
    x = _data(256)
    sae = TopKSAE(16, 32, k=4)
    sae.fit_scale(x)
    p = str(tmp_path / "s.pt")
    save_sae(sae, p)
    sae2 = load_sae(p)
    assert torch.equal(sae.encode(x), sae2.encode(x)) and sae2.scale == sae.scale


def test_is_massive_uses_norm_threshold_and_inf_means_none():
    sae = TopKSAE(16, 32, k=4)
    x = torch.randn(6, 16)
    x[2] *= 100
    assert not sae.is_massive(x).any()  # default threshold is inf
    sae.massive_threshold.fill_(5 * x.norm(dim=1).median().item())
    assert sae.is_massive(x).tolist() == [False, False, True, False, False, False]


def test_massive_threshold_is_saved_with_the_checkpoint(tmp_path):
    sae = TopKSAE(16, 32, k=4)
    sae.massive_threshold.fill_(123.0)
    save_sae(sae, str(tmp_path / "s.pt"))
    assert load_sae(str(tmp_path / "s.pt")).massive_threshold.item() == 123.0
