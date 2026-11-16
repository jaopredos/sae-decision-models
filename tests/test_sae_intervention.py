import torch
import torch.nn as nn

from src.sae import TopKSAE
from src.sae_intervention import SAEIntervention


class Tiny(nn.Module):
    """Three 'layers' over [B,S,H]; layer 1 is the hooked one."""
    def __init__(self, h=16):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(h, h) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


def _setup(seed=0):
    torch.manual_seed(seed)
    model = Tiny().eval()
    sae = TopKSAE(16, 40, k=4)
    x = torch.randn(3, 7, 16)
    h = model.layers[0](x)
    sae.fit_scale(h.reshape(-1, 16))
    return model, sae, x


def test_no_edit_is_exactly_identity():
    model, sae, x = _setup()
    base = model(x)
    with SAEIntervention(sae, model.layers[1], mode="edit"):
        assert torch.equal(model(x), base)


def test_edit_moves_output_only_through_the_edited_decoder_direction():
    model, sae, x = _setup()
    h1 = model.layers[1](torch.tanh(model.layers[0](x)))
    z = sae.encode(h1)
    feat = int(z.sum((0, 1)).argmax())
    with SAEIntervention(sae, model.layers[1], mode="edit", edits={feat: ("zero",)}) as iv:
        out_edit = {}
        h = iv.layer.register_forward_hook(lambda m, i, o: out_edit.setdefault("h", o.detach().clone()))
        model(x)
        h.remove()
    # the hook that edits runs first, so the second hook saw the edited tensor
    delta = out_edit["h"] - h1
    expected = -(z[..., feat:feat + 1] * sae.W_dec[feat]) / sae.scale
    assert torch.allclose(delta, expected, atol=1e-5)
    assert (delta.abs().sum(-1) > 0).any()


def test_position_mask_limits_the_edit():
    model, sae, x = _setup()
    base = model(x)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[0, 2] = True
    with SAEIntervention(sae, model.layers[1], mode="replace", position_mask=mask):
        out = model(x)
    h1 = torch.tanh(model.layers[0](x))
    changed = (model.layers[1](h1) != sae.reconstruct(model.layers[1](h1))).any(-1)  # sanity: recon differs
    assert changed.any()
    assert not torch.equal(out[0], base[0])
    assert torch.equal(out[1:], base[1:])


def test_clamp_turns_a_feature_on_everywhere():
    model, sae, x = _setup()
    captured = {}
    with SAEIntervention(sae, model.layers[1], mode="edit", edits={5: ("clamp", 3.0)}) as iv:
        hh = iv.layer.register_forward_hook(lambda m, i, o: captured.setdefault("h", o.detach().clone()))
        model(x)
        hh.remove()
    z_after = sae.encode(captured["h"])
    # after clamping the latent 5 is the largest or among the top-k at every position
    assert (z_after[..., 5] > 0).all()


def test_mean_and_zero_modes():
    model, sae, x = _setup()
    with SAEIntervention(sae, model.layers[1], mode="zero"):
        out_zero = model(x)
    with SAEIntervention(sae, model.layers[1], mode="mean", replacement=torch.zeros(16)):
        out_mean0 = model(x)
    assert torch.equal(out_zero, out_mean0)


def test_tuple_outputs_are_supported():
    class TupleLayer(nn.Module):
        def forward(self, x):
            return (x * 2, "aux")
    sae = TopKSAE(16, 40, k=4)
    sae.fit_scale(torch.randn(50, 16))
    layer = TupleLayer()
    x = torch.randn(2, 5, 16)
    with SAEIntervention(sae, layer, mode="zero"):
        out = layer(x)
    assert isinstance(out, tuple) and out[1] == "aux" and out[0].abs().sum() == 0


def test_massive_tokens_are_left_untouched_by_replace_and_zero():
    model, sae, x = _setup()
    h1 = model.layers[1](torch.tanh(model.layers[0](x)))
    big = (0, 3)
    h_big = h1.clone()
    h_big[big] = h_big[big] * 1000          # one attention-sink-like token
    sae.massive_threshold.fill_(5 * h1.norm(dim=-1).median().item())
    assert sae.is_massive(h_big)[big] and sae.is_massive(h_big).sum() == 1
    for mode in ("replace", "zero"):
        captured = {}
        with SAEIntervention(sae, model.layers[1], mode=mode) as iv:
            hh = iv.layer.register_forward_hook(lambda m, i, o: captured.setdefault("h", o.detach().clone()))
            iv.layer.register_forward_hook(lambda m, i, o: None)
            # feed a modified input so layer 1 emits the massive token
            model.layers[1].register_forward_hook(lambda m, i, o: h_big, prepend=True)
            model(x)
            hh.remove()
        # after the prepended hook swaps in h_big, the SAE hook ran on it: the massive token survives
        assert torch.equal(captured["h"][big], h_big[big])
        model.layers[1]._forward_hooks.clear()
