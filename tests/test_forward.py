import numpy as np

from ensiol import MTForward1D, SerialExecutor
from ensiol.forward import match_or_interpolate


def test_half_space_is_analytic(grid):
    app, ph = MTForward1D(grid.depths).response(np.full(grid.n_layers, 50.0), grid.freqs)
    assert np.allclose(app, 50.0, rtol=1e-6) and np.allclose(ph, 45.0, atol=1e-6)


def test_simulate_matches_single_responses(grid):
    fw = MTForward1D(grid.depths)
    models = np.exp(np.random.default_rng(0).normal(4, 1, (4, grid.n_layers)))
    app, ph = fw.simulate(models, grid.freqs, SerialExecutor())
    assert app.shape == (4, grid.freqs.size)
    a0, p0 = fw.response(models[2], grid.freqs)
    assert np.array_equal(app[2], a0) and np.array_equal(ph[2], p0)


def test_resampling_exact_and_interpolated(grid):
    f = grid.freqs
    A = np.log(np.outer(f, np.ones(3)) ** 0.5)
    P = np.outer(np.linspace(30, 60, f.size), np.ones(3))
    a, p, how, _ = match_or_interpolate(f, f[[1, 5, 7]], A, P)
    assert how == "exact" and np.array_equal(a, A[[1, 5, 7]])
    mid = np.sqrt(f[3] * f[4])
    a, p, how, _ = match_or_interpolate(f, np.array([mid]), A, P)
    assert how == "interp" and a.shape == (1, 3)
