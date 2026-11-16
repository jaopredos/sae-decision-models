import torch

from src.attribution import feature_attribution
from src.sae import TopKSAE


def _sae(d=16, f=48, k=5, seed=0):
    torch.manual_seed(seed)
    sae = TopKSAE(d, f, k=k)
    x = torch.randn(200, d) * 3 + 1
    sae.fit_scale(x)
    sae.init_b_dec(x)
    return sae, x


def test_attribution_sums_to_decoded_projection():
    sae, x = _sae()
    h, g = x[:30], torch.randn(30, 16)
    res = feature_attribution(sae, h, g)
    z = sae.encode(h)
    expect = ((sae.decode_rescaled(z) - sae.b_dec) / sae.scale * g).sum()
    assert torch.allclose(res["total_attr"], expect, atol=1e-4)


def test_ablating_a_feature_changes_a_linear_objective_by_minus_a():
    """delta(h) = w.h is linear, so removing feature i changes it by -a[t,i] exactly."""
    sae, x = _sae()
    h, w = x[:20], torch.randn(16)
    g = w.expand_as(h)
    res = feature_attribution(sae, h, g)
    z = sae.encode(h)
    t = 4
    for i in z[t].nonzero().flatten().tolist():
        h_abl = h[t] - z[t, i] * sae.W_dec[i] / sae.scale
        assert torch.allclose(w @ h_abl - w @ h[t], -res["a"][t, i], atol=1e-4)


def test_massive_tokens_get_no_attribution_and_do_not_enter_the_total():
    sae, x = _sae()
    h = x[:10].clone(); h[3] *= 1000
    sae.massive_threshold.fill_(5 * x.norm(dim=1).median().item())
    g = torch.randn(10, 16)
    res = feature_attribution(sae, h, g)
    assert res["massive"][3] and res["a"][3].abs().sum() == 0
    keep = ~res["massive"]
    assert torch.allclose(res["total_hg"], (h[keep] * g[keep]).sum(), atol=1e-2)
