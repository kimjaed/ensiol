import numpy as np

from ensiol import station_seed


def test_leakage_guard_removes_duplicates(ctx, prior):
    truth = prior.res[17]
    cat, n_removed = ctx.starting_catalog(truth)
    assert n_removed == 1 and cat.n == len(prior) - 1


def test_no_truth_keeps_everything(ctx, prior):
    cat, n_removed = ctx.starting_catalog()
    assert n_removed == 0 and cat.n == len(prior)


def test_station_seed_is_stable_and_distinct():
    assert station_seed(0, "keller") == station_seed(0, "keller")
    assert station_seed(0, "keller") != station_seed(0, "03_08ed")
    assert station_seed(5, "keller") == 5 + station_seed(0, "keller")
