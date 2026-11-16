import numpy as np
import torch
from scipy import stats

from src.features import bh, confusion_enrichment, enrichment_table, wilson


def _synthetic(n=600, f=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    acts = (torch.rand(n, f, generator=g) < 0.2).float() * torch.rand(n, f, generator=g)
    y = torch.rand(n, generator=g) < 0.5
    # feature 3 is strongly tied to the outcome, feature 7 to its absence
    y = y | ((acts[:, 3] > 0) & (torch.rand(n, generator=g) < 0.9))
    y = y & ~((acts[:, 7] > 0) & (torch.rand(n, generator=g) < 0.9))
    return acts, y


def test_one_sided_p_matches_scipy_fisher_exact():
    acts, y = _synthetic()
    t = enrichment_table(acts, y, min_support=5)
    a = (acts > 0).numpy(); yy = y.numpy()
    for f in (0, 3, 7, 12):
        table = [[int((a[:, f] & yy).sum()), int((a[:, f] & ~yy).sum())],
                 [int((~a[:, f] & yy).sum()), int((~a[:, f] & ~yy).sum())]]
        p_greater = stats.fisher_exact(table, alternative="greater")[1]
        p_less = stats.fisher_exact(table, alternative="less")[1]
        assert abs(t.loc[f, "p_more_y"] - p_greater) < 1e-9
        assert abs(t.loc[f, "p_less_y"] - p_less) < 1e-9


def test_the_planted_features_are_found():
    acts, y = _synthetic()
    t = enrichment_table(acts, y, min_support=20)
    assert t.loc[3, "q_bh"] < 0.01 and t.loc[3, "lift"] > 1.2
    assert t.loc[7, "q_bh"] < 0.05 and t.loc[7, "lift"] < 0.8
    noise = t.drop([3, 7])
    assert (noise["q_bh"] < 0.05).sum() <= 2


def test_support_threshold_gives_nan_p():
    acts = torch.zeros(100, 3); acts[:5, 0] = 1.0
    t = enrichment_table(acts, torch.rand(100) < 0.5, min_support=20)
    assert not t.loc[0, "supported"] and np.isnan(t.loc[0, "p_two_sided"]) and np.isnan(t.loc[0, "q_bh"])


def test_weighted_rate_uses_magnitudes():
    acts = torch.zeros(4, 1); acts[:, 0] = torch.tensor([1.0, 1.0, 1.0, 9.0])
    y = torch.tensor([False, False, False, True])
    t = enrichment_table(acts, y, min_support=1)
    assert abs(t.loc[0, "rate_active"] - 0.25) < 1e-6 and abs(t.loc[0, "weighted_rate"] - 0.75) < 1e-6


def test_wilson_matches_known_value_and_is_nan_without_support():
    lo, hi = wilson(np.array([8]), np.array([10]))
    assert abs(lo[0] - 0.4902) < 1e-3 and abs(hi[0] - 0.9433) < 1e-3
    assert np.isnan(wilson(np.array([0]), np.array([0]))[0][0])


def test_bh_matches_manual():
    p = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = bh(p)
    assert np.allclose(q[:4], [0.02, 0.04, 0.04, 0.02]) and np.isnan(q[4])


def test_confusion_enrichment_finds_the_planted_class():
    n = 400
    g = torch.Generator().manual_seed(1)
    acts = torch.zeros(n, 5)
    cls = torch.full((n,), -1)
    cls[:100] = torch.randint(0, 4, (100,), generator=g)
    acts[:, 2] = ((cls == 2) & (torch.rand(n, generator=g) < 0.9)).float()
    out = confusion_enrichment(acts, cls, n_classes=4, min_support=5)
    assert out.loc[2, "top_confusion_class"] == 2 and out.loc[2, "top_confusion_lift"] > 3
