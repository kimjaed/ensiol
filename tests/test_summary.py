import csv

import numpy as np

from ensiol.experiments import EnsIOLvsMCMC, SyntheticEvaluation
from ensiol.baselines import PCNConfig
from ensiol import ExperimentContext
from ensiol.summary import MetricsReport, SyntheticResults


def test_report_from_experiment_outputs(ctx, grid, prior, small_config, test_models, tmp_path):
    syn = SyntheticEvaluation(ctx, str(tmp_path / "syn"), n_rows=3, make_figures=False, workers=1)
    syn.run([0, 1, 2])
    syn.collect([0, 1, 2])
    c = ExperimentContext(grid, prior, small_config, pcn=PCNConfig(n_chains=8, max_calls=400), verbose=False)
    c.test = test_models
    mc = EnsIOLvsMCMC(c, str(tmp_path / "mc"), ensiol_dir=str(tmp_path / "syn" / "per_sample"),
                      ensiol_budgets="grid", make_figures=False, workers=1)
    mc.run([0, 1, 2])
    mc.collect([0, 1, 2])

    res = SyntheticResults.from_npz(str(tmp_path / "syn" / "all_samples.npz"))
    assert res.base.U.shape[0] == 3 and 0.0 <= res.aug.cov95 <= 1.0 and res.aug.Ubins is not None
    assert res.aug.nrms.mean() < res.base.nrms.mean()

    out = tmp_path / "tables"
    MetricsReport(str(tmp_path / "syn" / "all_samples.npz"), mcmc=str(tmp_path / "mc" / "results_long.csv"),
                  out_dir=str(out), echo=lambda s: None).run()
    rows = list(csv.reader(open(out / "table_synthetic.csv")))
    assert [r[0] for r in rows][:3] == ["Method", "Original catalog", "EnsIOL"] and len(rows) == 5
    assert (out / "table_synthetic.tex").read_text().startswith("\\begin{tabular}")
    assert (out / "summary_numbers.txt").exists()
