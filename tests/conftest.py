"""A tiny, fast synthetic setup shared by the tests."""
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from ensiol import (EnsIOLConfig, ExperimentContext, Grid, MTForward1D, Observation, PriorEnsemble,
                    SerialExecutor, SyntheticTestSet)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def grid():
    depths = np.concatenate([[0.0], np.geomspace(50.0, 8000.0, 9)])      # 9 layers, last is the half-space
    freqs = np.geomspace(1e-3, 1e2, 12)
    return Grid(depths, freqs)


@pytest.fixture(scope="session")
def small_config():
    return EnsIOLConfig(n_neighbors=60, n_neighbors_surrogate=60, n_add_max=10, max_iter=6, n_probe=100)


def _profiles(rng, n, n_layers):
    base = np.log(100.0) + np.cumsum(rng.normal(0, 0.35, size=(n, n_layers)), axis=1) * 0.5
    return np.exp(base - base.mean(axis=1, keepdims=True) + np.log(80.0))


@pytest.fixture(scope="session")
def prior(grid):
    rng = np.random.default_rng(0)
    res = _profiles(rng, 300, grid.n_layers)
    app, ph = grid.forward_operator().simulate(res, grid.freqs, SerialExecutor())
    return PriorEnsemble(res, app, ph)


@pytest.fixture(scope="session")
def test_models(grid):
    rng = np.random.default_rng(1)
    res = _profiles(rng, 3, grid.n_layers) * 2.0
    fw = grid.forward_operator()
    app, ph = fw.simulate(res, grid.freqs, SerialExecutor())
    app = np.exp(np.log(app) + 0.2 * rng.standard_normal(app.shape))
    ph = ph + 5.7 * rng.standard_normal(ph.shape)
    return SyntheticTestSet(res, app, ph, grid.freqs)


@pytest.fixture(scope="session")
def ctx(grid, prior, small_config, test_models):
    c = ExperimentContext(grid, prior, small_config, verbose=False)
    c.test = test_models
    return c
