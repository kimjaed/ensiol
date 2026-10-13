import numpy as np

from ensiol import Catalog, MinDistanceCalibrator, Whitener


def make_catalog(prior, grid):
    L, A, P = prior.arrays()
    return Catalog(L, A, P, grid.freqs)


def test_whitener_scales_by_std(prior):
    w = Whitener.from_labels(prior.labels)
    W = w.apply(prior.labels)
    assert W.shape == (len(prior), prior.res.shape[1])
    assert np.allclose(W.std(axis=0), 1.0)


def test_incremental_interpolation_matches_full(prior, grid):
    cat = make_catalog(prior, grid)
    obs_f = np.sqrt(grid.freqs[:-1] * grid.freqs[1:])          # forces the interpolation branch
    cat.interp_to_real(obs_f)
    L, A, P = prior.arrays()
    cat.append(L[:, :5], A[:, :5], P[:, :5])
    inc_A, inc_P = cat.interp_to_real(obs_f)
    fresh_A, fresh_P = make_catalog(prior, grid).with_additions(L, A, P, 5).interp_to_real(obs_f)
    assert np.allclose(inc_A, fresh_A) and np.allclose(inc_P, fresh_P)


def test_head_and_with_additions(prior, grid):
    cat = make_catalog(prior, grid)
    L, A, P = prior.arrays()
    grown = cat.with_additions(L, A, P, 7)
    assert grown.n == cat.n + 7 and grown.head(cat.n).n == cat.n


def test_min_distance_is_positive_and_deterministic(prior):
    w = Whitener.from_labels(prior.labels)
    cal = MinDistanceCalibrator(n_probe=100)
    d1, d2 = cal.calibrate(prior.labels, w), cal.calibrate(prior.labels, w)
    assert d1 > 0 and d1 == d2
