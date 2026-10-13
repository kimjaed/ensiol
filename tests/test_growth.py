import numpy as np

from ensiol import SerialExecutor


def run(ctx, idx=0):
    case = ctx.test.case(idx)
    return case, ctx.augment(case.observation, ctx.config.base_seed + idx, truth_linear=case.truth_linear)


def test_estimate_is_deterministic(ctx):
    case = ctx.test.case(0)
    cat, _ = ctx.starting_catalog()
    e1 = ctx.enscgp.estimate(case.observation, cat, SerialExecutor(), seed=3)
    e2 = ctx.enscgp.estimate(case.observation, cat, SerialExecutor(), seed=3)
    assert np.array_equal(e1.Labels_est, e2.Labels_est) and e1.n_neighbors_used == 60


def test_growth_adds_members_within_bounds_and_distance(ctx):
    case, r = run(ctx)
    assert r.n_added > 0 and r.n_added == int(r.growth.n_accepted.sum())
    lo, hi = ctx.config.log_bounds
    added = r.additions[0]
    assert added.min() >= lo and added.max() <= hi
    W_add, W_base = ctx.whitener.apply(added), ctx.whitener.apply(r.base.Labels)
    from scipy.spatial.distance import cdist
    assert cdist(W_add, W_base).min() > ctx.d_min - 1e-12


def test_growth_is_reproducible(ctx):
    _, a = run(ctx)
    _, b = run(ctx)
    assert np.array_equal(a.catalog.Labels, b.catalog.Labels)
    assert a.growth.exit_reason == b.growth.exit_reason


def test_cumulative_calls_equal_members_added(ctx):
    _, r = run(ctx)
    assert r.growth.cumulative_calls[-1] == r.n_added
    assert 0 <= r.growth.last_growing_iteration <= len(r.growth.history)


def test_augmentation_improves_misfit(ctx):
    case, r = run(ctx)
    rows = r.evolution_rows(iterations=[0, r.growth.last_growing_iteration])
    assert rows[-1]["nRMS"] < rows[0]["nRMS"]
    assert rows[-1]["catalog_size"] == r.catalog.n and rows[0]["n_added"] == 0
