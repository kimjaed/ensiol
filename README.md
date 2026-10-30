# EnsIOL: ensemble incremental online learning for geophysical inversion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23810947.svg)](https://doi.org/10.5281/zenodo.23810947)

Code accompanying the study of **EnsIOL**, a transfer scheme that grows a prior catalog of synthetic
(model, response) pairs against real observations from a new geologic setting, so that an
ensemble-based inversion (EnsCGP) can be moved from Gabbs Valley, Nevada to Kilauea, Hawaii without
retraining. The forward physics is 1-D magnetotellurics (MT).

The repository follows the layout of [`npi-mt`](https://github.com/kimjaed/npi-mt): an installable package
in `src/`, notebooks that reproduce the figures, and the scripts that run on the cluster.

## Quick start: from nothing to the manuscript figures

This walks through every step, assuming only that you have a terminal. Linux and macOS work directly; on Windows
use WSL. The whole sequence takes about 10 minutes plus a few minutes of computing.

### 1. Install conda (skip if you have it)

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and open a new terminal. Check with
`conda --version`. You also need `git` (`git --version`).

### 2. Create and activate an environment

```bash
conda create -n ensiol python=3.10 -y
conda activate ensiol
```

Run `conda activate ensiol` in every new terminal before using the code.

### 3. Get the code and install it

```bash
git clone https://github.com/kimjaed/ensiol.git
cd ensiol
pip install -e ".[notebooks,test]"
```

This installs the package `ensiol` (in editable mode, so the repository is the installation) with its three
dependencies (numpy, scipy, matplotlib) plus Jupyter and pytest. Everything below is run from this `ensiol/`
directory, the repository root.

An alternative that does steps 2 and 3 together is `conda env create -f environment.yml`, run from the repository
root, followed by `conda activate ensiol`.

### 4. Check the installation (about 10 seconds)

```bash
pytest
```

All tests should pass. If you see `ModuleNotFoundError: No module named 'ensiol'`, the environment is not
active or step 3 was not run in it.

### 5. Build the prior catalog (once)

The 10,000-member prior catalog is large, so it is not stored in git. It is rebuilt exactly from the Gabbs Valley
ensemble in `data/gabbs_valley/`:

```bash
python scripts/build_prior_catalog.py
```

This takes a minute or two and writes `kilauea_1d_{ResModels,AppRes,Phase}_prior.txt` into `data/kilauea/`.
Everything else the notebooks need (the 100 synthetic test cases, the grids and the Keller sounding) is already in
`data/`; see [`data/README.md`](data/README.md).

### 6. Run the example notebooks

```bash
jupyter lab notebooks/
```

Open a notebook, choose the kernel *Python 3* (the one from the `ensiol` environment), and use
*Run > Run All Cells*. Each notebook starts from the prior catalog and runs EnsIOL itself; nothing is loaded from
earlier runs. Figures are written to `figures/`.

| Order | Notebook | Reproduces | Computing time |
|---|---|---|---|
| 1 | [`01_fig2_evolution_sample14.ipynb`](notebooks/01_fig2_evolution_sample14.ipynb) | Fig. 2: ensemble of estimates at successive stages of augmentation, synthetic test case 14 | about 1 min |
| 2 | [`02_fig3_ensiol_vs_mcmc_sample14.ipynb`](notebooks/02_fig3_ensiol_vs_mcmc_sample14.ipynb) | Fig. 3: EnsIOL vs. MCMC and a naive grid sweep, accuracy and misfit against forward-model calls, test case 14 | 5-8 min (includes 20,000 MCMC forward calls and the simulation of the grids) |
| 3 | [`03_fig_real_station_keller.ipynb`](notebooks/03_fig_real_station_keller.ipynb) | Real data: ensembles at the Keller station before and after augmentation | about 1 min |

**What you should see.** The notebooks print the manuscript's numbers next to the ones they obtain, so you can
confirm the replication without looking at the figures.

* Notebook 1: the last row of the printed table: 1,931 models added, RMS 0.35 and nRMS 1.04 for test case 14.
* Notebook 2: EnsIOL 0.35 and 1.04 at 1,931 calls; MCMC 0.91 and 3.42 at 2,000 calls, and 0.67 and 1.92 at
  20,000 calls (RMS and nRMS, mean-of-responses convention). The grid sweep costs 1,430 forward calls at k = 7
  and 18,816 at k = 13 for every test case; the notebook prints these next to the manuscript values (`OK`),
  with its own RMS and nRMS for this case.
* Notebook 3: 21 frequencies, the catalog grows by about 1,140 members in 37 iterations, and nRMS falls from 2.77
  to 0.56 (the manuscript values). The minimum of the ensemble-mean resistivity is about 5.4 ohm-m in the layer
  between 1.155 and 1.243 km.

The figures land in `figures/`: `fig2_evolution_sample_014.{pdf,png}`, `fig3_ensiol_vs_mcmc_sample_014.{pdf,png}`,
and `fig_ensemble_keller_{before,after}.png` (plus `fig_evolution_keller.png`). Styling is controlled by the
`STYLE` objects in the notebooks (`PublicationStyle`, `ComparisonStyle`); change a setting with `STYLE.but(...)`
and rerun that cell.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: ...kilauea_1d_ResModels_prior.txt` | Step 5 was not run, or was run from another directory. Run it from the repository root. |
| `ModuleNotFoundError: No module named 'ensiol'` in a notebook | The notebook uses a different kernel. Start Jupyter from the activated `ensiol` environment and select *Python 3*. |
| Results differ slightly in the last digits | Expected across machines (BLAS rounding); the figures and the printed numbers agree to the digits shown. |
| A notebook is slow | Notebook 2 spends most of its time in the 20,000 MCMC forward calls; lower `MCMC_MAX_CALLS` for a quick look. The manuscript numbers need the full 20,000. |

## Reproducing the tables (cluster scale)

The notebooks cover single cases. Table I (100 synthetic test cases), the EnsIOL vs. MCMC comparison over all
cases, and the field results over all stations are cluster-scale computations, described next.

### Running the experiments

The scripts in [`scripts/`](scripts) run the full experiments. Each `.sh` is a SLURM job (one worker process per
sample or station, resumable: finished items are skipped on resubmission), submitted from the repository root.

| Step | Submit | Produces |
|---|---|---|
| 1. EnsIOL on the 100 synthetic test cases | `sbatch scripts/run_synthetic.sh` | `outputs/synthetic/` (per-sample results, `metrics.csv`, `all_samples.npz`, calibration) |
| 2. EnsIOL vs. MCMC catalog growth | `sbatch scripts/run_ensiol_vs_mcmc.sh` | `outputs/ensiol_vs_mcmc/` (reuses the catalogs from step 1) |
| 3. Naive grid sweep baseline | `sbatch scripts/run_pc_sweep_baseline.sh` | `outputs/pc_sweep/` (per-sample results, `results_long.csv`, `summary_by_grid.csv`, summary figure) |
| 4. Real Kilauea stations (those with at least 10 frequencies) | `sbatch scripts/run_real_data.sh` | `outputs/real_data/` |
| 5. Tables I and II and every number quoted in the text | `python scripts/summarize_metrics.py --mcmc outputs/ensiol_vs_mcmc/results_long.csv --pc-sweep outputs/pc_sweep/results_long.csv --stations outputs/real_data/station_summary.csv --per-station-dir outputs/real_data/per_station --depth-interfaces data/kilauea/kilauea_depth_interfaces_m.txt` | `outputs/tables/` (`table_synthetic.tex`, `table_real.tex`, `summary_numbers.txt`) |

Extra arguments are forwarded, e.g. `sbatch scripts/run_ensiol_vs_mcmc.sh --samples 14 --ensiol-budgets iterations --pc-grid`
scores EnsIOL after every iteration for one case and adds the grid sweep to that case's record.
`python scripts/<name>.py --help` lists all options.
Edit the Python path and e-mail lines in the `.sh` files for your account.

**Without a cluster.** The same scripts run directly and use all available cores, e.g.
`python scripts/run_synthetic.py --out-dir outputs/synthetic`. Finished samples are skipped when you rerun, so an
interrupted run just continues. On one core step 1 takes roughly half an hour (about 20 s per test case), whereas
step 2 takes a few minutes per test case because of the 20,000 MCMC forward calls, so it is worth parallelising.
`--samples 14` restricts any run to one test case. The grid sweep simulates its grids once (about a minute or two)
and then needs tens of seconds per sample.

**Station data.** `data/kilauea_stations/` ships only the Keller sounding. Step 4 needs one
`<station_id>_1d_response.dat` file per station (columns: frequency in Hz, apparent resistivity in ohm-m, phase in
degrees); put the others there to run the full field application.

## How the code is organised

```
src/ensiol/
  config.py       NoiseModel, EnsIOLConfig, ScoringConfig      all hyperparameters, frozen dataclasses
  forward.py      MTForward1D, SerialExecutor                  1-D MT forward operator, frequency resampling
  data.py         Grid, Observation, PriorEnsemble,            data containers and loaders
                  SyntheticTestSet, RealStation, KilaueaDataset
  catalog.py      Catalog, Whitener, MinDistanceCalibrator     the growing catalog and its geometry
  enscgp.py       EnsCGP, Estimate                             backward update and the evaluation estimate
  growth.py       EnsIOL, CandidateSelector,                   the growth loop and its parts
                  SurrogateCorrector, GrowthResult
  baselines.py    PCNSampler, PCNConfig,                       baselines for catalog growth: MCMC and the
                  PCGridSweep, PCGridConfig, PCBasis           naive principal-component grid sweep
  metrics.py      Scorer, rms_log10, coverage_tables           accuracy, misfit, calibration
  summary.py      MetricsReport, SyntheticResults, ...         Tables I and II and the numbers quoted in the text
  snapshots.py    SnapshotEvaluator                            the estimate at successive stages of growth
  context.py      ExperimentContext, AugmentationRun           wires the components; runs EnsIOL for one observation
  experiments/    Experiment (base), SyntheticEvaluation,      the cluster experiments, built on one base class
                  RealDataApplication, EnsIOLvsMCMC,
                  PCSweepBaseline
  priors/         GabbsValleyStatistics, GMRFPrior,            builds the prior catalog and the test models
                  NewSettingSpec
  plotting/       figure functions (return figures, never show)
scripts/          cluster entry points and data-build scripts
notebooks/        the figure notebooks
tests/            unit tests (pytest)
```

`ExperimentContext` is the entry point: it holds the grid, the prior catalog, the whitening and the calibrated
minimum distance, and builds every component from them.

```python
from ensiol import ExperimentContext, Observation
ctx = ExperimentContext.from_data_dir("data/kilauea", with_test=True)
case = ctx.test.case(14)
run = ctx.augment(case.observation, seed=14, truth_linear=case.truth_linear)   # EnsIOL on one observation
rows = run.evolution_rows(n_rows=5)       # single backward EnsCGP updates at successive stages
```

## Conventions

* **Estimate.** Every reported estimate is a single backward EnsCGP update on the 500 catalog members closest to the
  observation in data space (damping 1). The forward/backward correction exists only inside the EnsIOL growth loop,
  where a surrogate response built from the catalog is used for the correction and the true forward physics for
  adding members.
* **RMS** is the RMS error of the ensemble-mean log10 resistivity against the truth, with the half-space (last layer)
  excluded (`ScoringConfig.exclude_halfspace`).
* **nRMS** is the misfit of the *mean of the members' simulated responses* (`MoR`). The misfit of the response of the
  mean model (`RoM`) is always computed and stored as well (`ScoringConfig.nrms_convention`).
* **Noise** is 10% relative error: 0.2 on log apparent resistivity and 5.71 degrees on phase (`NoiseModel`).
* **Cost** of augmenting a catalog is the number of forward-model calls made to build it: one per added member for
  EnsIOL, the total number of sampler calls for MCMC, and the number of grid models inside the physical bounds for
  the grid sweep.
* **Grid sweep** is the naive baseline. It uses no information from the observation: a regular grid of *k* values per
  axis over the four leading principal components of the original catalog (the catalog mean plus the four fixed depth
  patterns scaled by coefficients, each coefficient ranging between its smallest and largest value in the catalog),
  giving *k*⁴ models, of which those outside the physical bounds are skipped without a forward call. Each grid size is a
  separate catalog built from the original one, scored by the same single backward EnsCGP update as the other methods.
* **Seeds.** Synthetic sample *i* uses seed `base_seed + i`. A station uses `base_seed + crc32(station_id)`
  (deterministic across processes, unlike Python's `hash`).

## Tests

```bash
pytest
```

## Citing

If you use this software in academic work, please cite it using [`CITATION.cff`](CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
