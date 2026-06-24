import numpy as np
from src.utils import make_nested_subsampler

def _make_y(n_pos, n_neg):
    return np.array([1] * n_pos + [0] * n_neg)

def test_returns_requested_size_approximately():
    y = _make_y(300, 700)  # 1000, 30% positivi
    sub = make_nested_subsampler(y, seed=12)
    idx = sub(100)
    assert abs(len(idx) - 100) <= 1

def test_stratified_proportions_preserved():
    y = _make_y(300, 700)  # 30% positivi
    sub = make_nested_subsampler(y, seed=12)
    idx = sub(200)
    frac_pos = y[idx].mean()
    assert abs(frac_pos - 0.3) < 0.03

def test_nested_subsamples():
    y = _make_y(300, 700)
    sub = make_nested_subsampler(y, seed=12)
    small = set(sub(100).tolist())
    big = set(sub(400).tolist())
    assert small.issubset(big)

def test_deterministic_with_same_seed():
    y = _make_y(300, 700)
    a = make_nested_subsampler(y, seed=12)(250)
    b = make_nested_subsampler(y, seed=12)(250)
    assert np.array_equal(a, b)

def test_caps_at_pool_size():
    y = _make_y(300, 700)
    sub = make_nested_subsampler(y, seed=12)
    idx = sub(100000)  # oltre il pool
    assert len(idx) == 1000
    assert sorted(idx.tolist()) == list(range(1000))
