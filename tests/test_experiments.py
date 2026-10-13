import os

import numpy as np

from ensiol.experiments import EnsIOLvsMCMC, SyntheticEvaluation
from ensiol.baselines import PCNConfig
from ensiol import ExperimentContext


def test_synthetic_run_resume_and_collect(ctx, tmp_path):
    exp = SyntheticEvaluation(ctx, str(tmp_path), n_rows=3, make_figures=False, workers=1)
    assert exp.run([0, 1]) == []
    assert all(os.path.exists(exp.item_path(i)) for i in (0, 1))
    assert not list(tmp_path.glob("per_sample/*.tmp.npz"))
    assert exp.pending([0, 1, 2]) == [2]                    # finished samples are skipped
    exp.collect([0, 1])
    S = np.load(tmp_path / "all_samples.npz", allow_pickle=True)
    assert S["base_RMS_log10"].shape == (2,) and (tmp_path / "calibration_summary.csv").exists()
    assert np.all(S["aug_nRMS_MoR"] < S["base_nRMS_MoR"])
    assert np.all(S["base_RMS_log10"] == S["base_RMS_log10_excl"])      # manuscript convention is the default


def test_ensiol_vs_mcmc_modes(ctx, grid, prior, small_config, test_models, tmp_path):
    c = ExperimentContext(grid, prior, small_config, pcn=PCNConfig(n_chains=8, max_calls=200), verbose=False)
    c.test = test_models
    exp = EnsIOLvsMCMC(c, str(tmp_path / "iter"), ensiol_budgets="iterations", make_figures=False, workers=1)
    rec = exp.run_item(0)
    assert rec["budgets_ensiol"][0] == 0 and rec["budgets_ensiol"][-1] == rec["ensiol_total_calls"]
    assert rec["mcmc_total_calls"] >= 200 and rec["ensiol_calls"][0] == 0
    grid_exp = EnsIOLvsMCMC(c, str(tmp_path / "grid"), ensiol_budgets="grid", make_figures=False, workers=1)
    rec = grid_exp.run_item(0)
    assert np.array_equal(rec["budgets_ensiol"], rec["budgets_mcmc"])


def test_figures_are_written_when_requested(ctx, grid, prior, small_config, test_models, tmp_path):
    c = ExperimentContext(grid, prior, small_config, pcn=PCNConfig(n_chains=8, max_calls=120), verbose=False)
    c.test = test_models
    exp = EnsIOLvsMCMC(c, str(tmp_path), ensiol_budgets="grid", make_figures=True, workers=1)
    exp.run([0])
    assert (tmp_path / "trajectory_figs" / "accuracy_vs_cost_sample_000.png").exists()
    syn = SyntheticEvaluation(ctx, str(tmp_path / "syn"), n_rows=3, make_figures=True, workers=1)
    syn.run([0])
    assert (tmp_path / "syn" / "evolution_figs" / "fig_synthetic_evolution_sample_000.png").exists()
