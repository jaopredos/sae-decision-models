import numpy as np
import torch

from run_ablation import Runner, effect, margin


def test_pred_marker_mask_marks_exactly_the_predicted_options_marker_across_chunks():
    r = Runner.__new__(Runner)  # only the mask logic is under test
    r.cur, r.count = {}, 0
    r.pred0 = torch.tensor([1, 0, 2, 1])             # baseline pick per global row
    marker_pos = torch.tensor([[3, 7, 9], [2, 5, 8], [1, 4, 6], [2, 6, 9]])
    for rows in (slice(0, 2), slice(2, 4)):          # two chunks of two rows
        r._pre(None, (torch.zeros(2, 12, dtype=torch.long), None, marker_pos[rows]))
        h = torch.zeros(2, 12, 4)
        m = r.pred_marker(h)
        for i, g in enumerate(range(rows.start, rows.stop)):
            assert m[i].nonzero().flatten().tolist() == [marker_pos[g, r.pred0[g]].item()]


def test_effect_margin_and_fixed_broken_are_consistent():
    base = np.array([[0., 2., 0., 0., 0.], [3., 0., 0., 0., 0.]])
    gold = np.array([0, 0])                           # row 0 is an error (picks 1), row 1 is right
    L = np.array([[2.5, 2., 0., 0., 0.], [1., 3., 0., 0., 0.]])  # row 0 fixed, row 1 broken
    m, dm, err0 = effect(L, base, gold)
    assert err0.tolist() == [True, False]
    assert m["fixed"] == 1.0 and m["broken"] == 1.0 and m["acc"] == 0.5
    assert np.isclose(dm[0], (2.5 - 2.0) - (0 - 2.0)) and np.isclose(m["dm_err"], 2.5)
    assert margin(base, base.argmax(1), gold).tolist() == [-2.0, 0.0]
