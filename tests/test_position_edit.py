import numpy as np
import torch
import torch.nn as nn

from src.position_edit import (CaptureMarkers, ProjectOut, ReplaceMarkers, ZeroHeads, between_class_directions,
                               random_subspace)


def test_between_class_directions_recovers_the_planted_subspace():
    rng = np.random.default_rng(0)
    d = 32
    planted = np.linalg.qr(rng.standard_normal((d, 3)))[0]
    codes = rng.standard_normal((4, 3)) * 3          # 4 classes -> centred offsets of rank 3
    X, grp, cls = [], [], []
    for g in range(6):
        base = rng.standard_normal(d) * 5
        for c in range(4):
            for _ in range(30):
                X.append(base + planted @ codes[c] + rng.standard_normal(d) * 0.05)
                grp.append(g); cls.append(c)
    U, r, expl = between_class_directions(np.array(X), np.array(grp), np.array(cls), energy=0.999)
    assert r == 3 and expl[-1] >= 0.999
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-5)
    assert np.linalg.norm(planted - U @ (U.T @ planted)) < 0.02 * np.linalg.norm(planted)  # planted lies in span(U)


def test_random_subspace_is_orthonormal():
    U = random_subspace(16, 4, np.random.default_rng(1))
    assert U.shape == (16, 4) and np.allclose(U.T @ U, np.eye(4), atol=1e-5)


def test_project_out_only_at_masked_positions_and_is_idempotent():
    torch.manual_seed(0)
    layer = nn.Identity()
    h = torch.randn(2, 5, 8)
    U = torch.linalg.qr(torch.randn(8, 2))[0].numpy()
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, [1, 3]] = True
    with ProjectOut(layer, U, lambda x: mask) as _:
        out = layer(h)
    Ut = torch.from_numpy(U)
    assert torch.allclose(out[mask] @ Ut, torch.zeros(4, 2), atol=1e-5)
    assert torch.equal(out[~mask], h[~mask])
    assert torch.allclose(out[mask], h[mask] - (h[mask] @ Ut) @ Ut.T, atol=1e-6)
    with ProjectOut(layer, U) as _:
        twice = layer(layer(h))
    assert torch.allclose(twice, layer(h) - (h @ Ut) @ Ut.T, atol=1e-5)


def test_projection_handles_tuple_outputs_and_keeps_dtype():
    class Tup(nn.Module):
        def forward(self, x):
            return (x, "aux")
    h = torch.randn(1, 3, 4, dtype=torch.float16)
    with ProjectOut(Tup(), np.eye(4, 1, dtype=np.float32)) as _:
        pass
    m = Tup()
    with ProjectOut(m, np.eye(4, 1, dtype=np.float32)) as _:
        out = m(h)
    assert isinstance(out, tuple) and out[1] == "aux" and out[0].dtype == torch.float16 and torch.all(out[0][..., 0] == 0)


def test_capture_then_replace_markers_round_trips_by_row_and_slot():
    layer = nn.Identity()
    pos = torch.tensor([[1, 3, -1], [0, 2, 4]])
    h = torch.arange(2 * 6 * 2, dtype=torch.float32).view(2, 6, 2)
    cap = CaptureMarkers(layer, lambda: pos)
    with cap:
        layer(h)
    got = cap.result()
    assert torch.equal(got[0, 1], h[0, 3]) and torch.equal(got[1, 2], h[1, 4])
    valid = pos >= 0
    swap = got.flip(1)  # reversed slots
    with ReplaceMarkers(layer, lambda: (pos, swap, valid)) as _:
        out = layer(torch.zeros_like(h))
    assert torch.equal(out[1, 0], got[1, 2]) and torch.equal(out[1, 4], got[1, 0])
    assert torch.equal(out[0, 1], got[0, 2 - 1 + 0] if False else swap[0, 0])
    assert out[0, 5].abs().sum() == 0 and out[0, 2].abs().sum() == 0  # untouched positions stay as the input (zeros)


def test_zero_heads_zeroes_exactly_the_requested_slices():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.Wo = nn.Linear(8, 8, bias=False)
            nn.init.eye_(self.Wo.weight)
    a = Attn()
    x = torch.ones(2, 3, 8)
    with ZeroHeads(a, [1, 3], n_heads=4) as _:
        y = a.Wo(x)
    assert torch.equal(y[..., 2:4], torch.zeros(2, 3, 2)) and torch.equal(y[..., 6:8], torch.zeros(2, 3, 2))
    assert torch.equal(y[..., 0:2], torch.ones(2, 3, 2)) and torch.equal(y[..., 4:6], torch.ones(2, 3, 2))
    assert torch.equal(a.Wo(x), x)  # hook removed
