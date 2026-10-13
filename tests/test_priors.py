import numpy as np

from conftest import REPO
from ensiol.priors import GabbsValleyStatistics, GMRFPrior, NewSettingSpec, smoothing_operator


def test_smoothing_operator_rows_sum_to_one():
    K = smoothing_operator(20, 2.0)
    assert np.allclose(K.sum(axis=1), 1.0)


def test_new_setting_samples_are_reproducible_and_bounded():
    depths = np.loadtxt(REPO / "data/kilauea/kilauea_depth_interfaces_m.txt")
    spec = NewSettingSpec()
    a1, h1 = spec.sample(depths, 6, 3, seed=2026)
    a2, h2 = spec.sample(depths, 6, 3, seed=2026)
    assert a1.shape == (6, depths.size - 1) and h1.shape == (3, depths.size - 1)
    assert np.array_equal(a1, a2) and not np.array_equal(a1[:3], h1)
    assert a1.min() >= spec.res_bounds[0] and a1.max() <= spec.res_bounds[1]


def test_gmrf_prior_from_gabbs_valley_is_reproducible():
    depths = np.loadtxt(REPO / "data/kilauea/kilauea_depth_interfaces_m.txt")
    stats = GabbsValleyStatistics.from_file(str(REPO / "data/gabbs_valley/real_data_ensemble.npy"))
    prior = GMRFPrior(stats, depths, verbose=False)
    a, b = prior.sample(50, seed=10), prior.sample(50, seed=10)
    assert a.shape == (50, depths.size - 1) and np.array_equal(a, b)
    assert np.allclose(prior.modes.T @ prior.modes, np.eye(prior.modes.shape[1]), atol=1e-10)
