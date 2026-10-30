"""Tests of the naive principal-component grid sweep and everything built on it."""
import csv
from dataclasses import replace

import numpy as np
import pytest

from ensiol import ExperimentContext, PCGridConfig, PCNConfig, SerialExecutor
from ensiol.experiments import EnsIOLvsMCMC, PCSweepBaseline, SyntheticEvaluation
from ensiol.experiments.mcmc_comparison import cost_series
from ensiol.plotting import ComparisonStyle, plot_comparison
from ensiol.summary import MetricsReport

PC = PCGridConfig(n_modes=2, grid_sizes=(3, 4))          # small: 9 and 16 nominal grid models


@pytest.fixture(scope="module")
def ctx_pc(grid, prior, small_config, test_models):
    c = ExperimentContext(grid, prior, small_config, pc_grid=PC, verbose=False)
    c.test = test_models
    return c


def test_basis_ranges_come_from_the_catalog(ctx_pc):
    b = ctx_pc.pc_sweep.basis
    labels = ctx_pc.prior.labels
    assert b.n_modes == 2 and np.allclose(b.U.T @ b.U, np.eye(2))
    coef = b.U.T @ (labels - b.mu)
    assert np.allclose(coef.min(axis=1), b.cmin) and np.allclose(coef.max(axis=1), b.cmax)
    assert 0 < b.var_explained <= 1


def test_grid_cost_is_the_number_of_models_inside_the_bounds(ctx_pc):
    sw = ctx_pc.pc_sweep
    sw.build_grids(verbose=False)
    lo, hi = ctx_pc.config.log_bounds
    for k in PC.grid_sizes:
        g = sw.grids[k]
        assert g.n_nominal == k ** 2                                   # every combination of the coefficients
        assert g.n_valid <= g.n_nominal and g.calls == g.n_valid
        assert g.models.min() >= lo and g.models.max() <= hi
        assert g.app_res_log.shape == (ctx_pc.grid.freqs.size, g.n_valid) and g.phase.shape == g.app_res_log.shape


def test_models_outside_the_bounds_are_dropped_without_simulation(grid, prior, small_config):
    tight = replace(small_config, res_min=60.0, res_max=110.0)
    c = ExperimentContext(grid, prior, tight, pc_grid=PC, verbose=False)
    c.pc_sweep.build_grids(verbose=False)
    lo, hi = tight.log_bounds
    for g in c.pc_sweep.grids.values():
        assert g.n_valid <= g.n_nominal
        assert g.models.size == 0 or (g.models.min() >= lo and g.models.max() <= hi)


def test_models_too_close_to_the_catalog_are_not_admitted(ctx_pc):
    sw = ctx_pc.pc_sweep
    base, _ = ctx_pc.starting_catalog()
    W = ctx_pc.whitener.apply(base.Labels)
    assert not sw._admitted(W[:5] + 1e-6, W).any()               # essentially duplicates of members
    assert sw._admitted(W[:5] + 1e3, W).all()                    # far from everything


def test_evaluation_is_deterministic_and_does_not_touch_the_catalog(ctx_pc):
    sw = ctx_pc.pc_sweep
    case = ctx_pc.test.case(0)
    base, _ = ctx_pc.starting_catalog(case.truth_linear)
    n0 = base.n
    r1 = sw.evaluate(base, case.observation, case.truth_log10, seed=0)
    r2 = sw.evaluate(base, case.observation, case.truth_log10, seed=0)
    assert base.n == n0
    assert list(r1["grid_sizes"]) == [3, 4]
    for key in ("rms", "nrms_rom", "nrms_mor", "U", "cov95"):
        assert np.array_equal(r1[f"grid_{key}"], r2[f"grid_{key}"]) and np.all(np.isfinite(r1[f"grid_{key}"]))
    assert np.all(r1["grid_added"] <= r1["grid_calls"])
    assert np.all((0 <= r1["grid_nbr_share"]) & (r1["grid_nbr_share"] <= 1))
    assert np.all(r1["grid_U"] >= 1) and np.all((0 <= r1["grid_cov95"]) & (r1["grid_cov95"] <= 1))
    # the grid never looks at the observation: another sample is charged the same forward calls
    other = ctx_pc.test.case(1)
    base1, _ = ctx_pc.starting_catalog(other.truth_linear)
    r3 = sw.evaluate(base1, other.observation, other.truth_log10, seed=1)
    assert np.array_equal(r1["grid_calls"], r3["grid_calls"])


def test_scorer_reports_spread_metrics(ctx_pc):
    case = ctx_pc.test.case(0)
    base, _ = ctx_pc.starting_catalog(case.truth_linear)
    s = ctx_pc.scorer.score_catalog(base, case.observation, case.truth_log10, 0, SerialExecutor())
    assert s["U"] >= 1 and 0 <= s["cov95"] <= 1 and np.isfinite(s["rms"])


def test_experiment_run_resume_and_collect(ctx_pc, tmp_path):
    exp = PCSweepBaseline(ctx_pc, str(tmp_path), make_figures=True, workers=1)
    assert exp.run([0, 1]) == []
    assert exp.pending([0, 1, 2]) == [2]                                   # finished samples are skipped
    other = PCSweepBaseline(ctx_pc, str(tmp_path), grid_sizes=[3], workers=1)
    assert other.pending([0, 1]) == [0, 1]                                 # different grid sizes: recompute
    exp.collect([0, 1])
    rows = list(csv.DictReader(open(tmp_path / "results_long.csv")))
    assert len(rows) == 2 * len(PC.grid_sizes) and {"U", "cov95", "RMS_original"} <= set(rows[0])
    assert (tmp_path / "summary_by_grid.csv").exists() and (tmp_path / "pc_grid_accuracy_vs_cost.png").exists()


def test_comparison_record_and_figure_include_the_grid(grid, prior, small_config, test_models, tmp_path):
    c = ExperimentContext(grid, prior, small_config, pcn=PCNConfig(n_chains=8, max_calls=200), pc_grid=PC, verbose=False)
    c.test = test_models
    exp = EnsIOLvsMCMC(c, str(tmp_path), ensiol_budgets="iterations", make_figures=False, pc_grid=True, workers=1)
    rec = exp.run_item(0)
    assert list(rec["pcgrid_sizes"]) == [3, 4] and "pcgrid_rms" in rec and "pcgrid_nrms_mor" in rec
    for key in ("U", "cov95"):
        assert f"ensiol_{key}" in rec and f"mcmc_catalog_{key}" in rec
    args = (cost_series(rec, "ensiol", "iterations"), cost_series(rec, "mcmc"), int(rec["ensiol_total_calls"]),
            int(rec["mcmc_n_added_total"]), int(rec["mcmc_total_calls"]))
    fig = plot_comparison(*args, ComparisonStyle(), "nrms_mor", grid=cost_series(rec, "pcgrid"))
    assert any(l.get_color() == "tab:orange" for l in fig.axes[0].lines)
    fig = plot_comparison(*args, ComparisonStyle().but(show_pc=False), "nrms_mor", grid=cost_series(rec, "pcgrid"))
    assert not any(l.get_color() == "tab:orange" for l in fig.axes[0].lines)


def test_table_includes_grid_rows_and_mcmc_spread(ctx_pc, grid, prior, small_config, test_models, tmp_path):
    syn = SyntheticEvaluation(ctx_pc, str(tmp_path / "syn"), n_rows=3, make_figures=False, workers=1)
    syn.run([0, 1, 2])
    syn.collect([0, 1, 2])
    c = ExperimentContext(grid, prior, small_config, pcn=PCNConfig(n_chains=8, max_calls=400), verbose=False)
    c.test = test_models
    mc = EnsIOLvsMCMC(c, str(tmp_path / "mc"), ensiol_dir=str(tmp_path / "syn" / "per_sample"),
                      ensiol_budgets="grid", make_figures=False, workers=1)
    mc.run([0, 1, 2])
    mc.collect([0, 1, 2])
    pcs = PCSweepBaseline(ctx_pc, str(tmp_path / "pc"), make_figures=False, workers=1)
    pcs.run([0, 1, 2])
    pcs.collect([0, 1, 2])

    out = tmp_path / "tables"
    MetricsReport(str(tmp_path / "syn" / "all_samples.npz"), mcmc=str(tmp_path / "mc" / "results_long.csv"),
                  out_dir=str(out), echo=lambda s: None, pc_sweep=str(tmp_path / "pc" / "results_long.csv"),
                  pc_sizes=(3, 4)).run()
    rows = list(csv.reader(open(out / "table_synthetic.csv")))
    labels = [r[0] for r in rows]
    assert "Grid sweep (k=3)" in labels and "Grid sweep (k=4)" in labels
    mcmc_row = next(r for r in rows if r[0].startswith("MCMC catalog, "))
    assert mcmc_row[5] != "--" and mcmc_row[6].endswith("%")              # U and coverage now reported
