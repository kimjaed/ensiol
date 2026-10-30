"""Naive baseline for catalog growth: a regular grid over the leading principal components.

The baseline uses no information from the observation. It varies the ``n_modes`` leading principal
components of the ORIGINAL catalog (natural-log resistivity) on a regular grid of ``k`` equally spaced
values per component, between the smallest and the largest coefficient the catalog itself has on that
component (endpoints included), and maps every combination back to a 54-layer model,

    m = mu + U a,

with ``mu`` the catalog mean and ``a`` the coefficients. All ``k ** n_modes`` combinations are used.
Models outside the physical bounds are discarded WITHOUT a forward call; every other model costs
exactly one. Each grid size is its own point on the cost curve, because a grid with k points per axis
is not a prefix of a grid with more points; the output has one row per (sample, k).

The grid does not depend on the observation, so every grid is simulated once in the main process and
shared with the workers (fork). Each sample is still charged the full forward calls of every grid, as
if it had run the sweep itself. Per sample and grid size, a grid model joins the catalog if it is
farther than ``d_min`` (whitened distance) from every member of the sample's starting catalog, and the
grown catalog is scored by the same single backward EnsCGP update as every other method.

Diagnostics stored per (sample, k): forward calls, models added to the catalog, and the share of the
500 nearest neighbours used by the evaluation update that come from the sweep.

Usage (from the repository root):
  python scripts/run_pc_sweep_baseline.py                           # all test samples
  python scripts/run_pc_sweep_baseline.py --samples 14 0            # quick check
  python scripts/run_pc_sweep_baseline.py --grid-sizes 6 7 12       # other grid sizes
  python scripts/run_pc_sweep_baseline.py --collect-only            # rebuild summaries and figure
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ..baselines import PCGridConfig
from ..context import ExperimentContext
from ..forward import SerialExecutor
from .base import Experiment

RESULT_KEYS = ("rms", "nrms_rom", "nrms_mor", "U", "cov95")


class PCSweepBaseline(Experiment):
    """One result file per test sample, holding one score per grid size.

    ``compare_dir`` is the output folder of :class:`EnsIOLvsMCMC` (optional); when it exists, the
    summary figure adds EnsIOL and MCMC for the same samples.
    """

    def __init__(self, ctx: ExperimentContext, out_dir: str, grid_sizes: Optional[Sequence[int]] = None,
                 compare_dir: Optional[str] = None, make_figures: bool = True, **kw):
        super().__init__(out_dir, **kw)
        assert ctx.test is not None, "context was built without a test set"
        self.ctx, self.sweep = ctx, ctx.pc_sweep
        self.grid_sizes: List[int] = [int(k) for k in (grid_sizes or ctx.pc_grid.grid_sizes)]
        self.compare_dir, self.make_figures = compare_dir, make_figures
        os.makedirs(f"{out_dir}/per_sample", exist_ok=True)

    # ---- bookkeeping -------------------------------------------------------------
    def item_path(self, idx) -> str:
        return f"{self.out_dir}/per_sample/sample_{idx:03d}.npz"

    def _done(self, idx) -> bool:
        """A sample counts as done only if it was computed for the same grid sizes."""
        f = self.item_path(idx)
        if not os.path.exists(f):
            return False
        try:
            with np.load(f, allow_pickle=True) as z:
                return "grid_rms" in z.files and list(z["grid_sizes"]) == self.grid_sizes
        except Exception:
            return False

    def pending(self, items: Sequence[int]) -> List[int]:
        return [i for i in items if not (self.resume and self._done(i))]

    def prepare(self, todo: Sequence[int]) -> None:
        print(self.sweep.describe_basis(), flush=True)
        print("simulating the grids (once, shared by all samples):", flush=True)
        self.sweep.build_grids(self.grid_sizes, SerialExecutor())

    # ---- one sample -----------------------------------------------------------------
    def run_item(self, idx: int) -> dict:
        t0 = time.time()
        ctx = self.ctx
        case = ctx.test.case(idx)
        seed = ctx.config.base_seed + idx
        base, _ = ctx.starting_catalog(case.truth_linear)                      # leakage guard
        r = self.sweep.evaluate(base, case.observation, case.truth_log10, seed, SerialExecutor(), self.grid_sizes)
        return dict(sample_idx=idx, seed=seed, truth_log10=case.truth_log10, runtime_s=time.time() - t0, **r)

    def summary_line(self, r: dict) -> str:
        ks = [int(k) for k in r["grid_sizes"]]
        return (f"sample {int(r['sample_idx']):3d}  RMS original {float(r['orig_rms']):.3f} | grid "
                + " ".join(f"k{k}:{v:.3f}" for k, v in zip(ks, r["grid_rms"])) + f"  ({float(r['runtime_s']):.0f} s)")

    # ---- aggregation ----------------------------------------------------------------
    def _load_compare(self, sample_indices) -> List[dict]:
        """EnsIOL and MCMC results of the comparison run for the same samples, if present."""
        recs = []
        if not self.compare_dir:
            return recs
        for i in sample_indices:
            f = f"{self.compare_dir}/per_sample/sample_{i:03d}.npz"
            if os.path.exists(f):
                try:
                    recs.append(dict(np.load(f, allow_pickle=True)))
                except Exception:
                    pass
        return recs

    def collect(self, items: Sequence[int]) -> None:
        ks = self.grid_sizes
        recs = [dict(np.load(self.item_path(i), allow_pickle=True)) for i in items if self._done(i)]
        if not recs:
            print("no finished samples to collect", flush=True)
            return
        print(f"\ncollected {len(recs)} samples", flush=True)
        stack = lambda key: np.array([r[key] for r in recs])
        rms0, nr0 = stack("orig_rms"), stack("orig_nrms_mor")
        out = self.out_dir

        with open(f"{out}/results_long.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "grid_size_k", "forward_calls", "catalog_added", "nbr_share", "RMS_log10",
                        "nRMS_RoM", "nRMS_MoR", "U", "cov95", "RMS_original", "nRMS_MoR_original"])
            for r in recs:
                for j, k in enumerate(ks):
                    w.writerow([int(r["sample_idx"]), k, int(r["grid_calls"][j]), int(r["grid_added"][j]),
                                r["grid_nbr_share"][j], r["grid_rms"][j], r["grid_nrms_rom"][j],
                                r["grid_nrms_mor"][j], r["grid_U"][j], r["grid_cov95"][j],
                                float(r["orig_rms"]), float(r["orig_nrms_mor"])])

        print(f"\noriginal catalog: RMS mean {rms0.mean():.3f} median {np.median(rms0):.3f} | "
              f"nRMS mean {nr0.mean():.2f} median {np.median(nr0):.2f}")
        print(f"{'k':>3s} {'calls':>7s} {'added':>7s} {'nbr%':>5s}  {'RMS mean':>9s} {'median':>7s}  "
              f"{'nRMS mean':>10s} {'median':>7s}  {'U':>5s} {'cov95':>6s}  {'RMS better':>10s}")
        summary = []
        for j, k in enumerate(ks):
            rms, nr = stack("grid_rms")[:, j], stack("grid_nrms_mor")[:, j]
            uu, cc = stack("grid_U")[:, j], stack("grid_cov95")[:, j]
            calls, add, sh = stack("grid_calls")[:, j], stack("grid_added")[:, j], stack("grid_nbr_share")[:, j]
            better = float(np.mean(rms < rms0))
            summary.append([k, calls.mean(), add.mean(), 100 * sh.mean(), rms.mean(), np.median(rms),
                            nr.mean(), np.median(nr), uu.mean(), cc.mean(), 100 * better])
            print(f"{k:3d} {calls.mean():7.0f} {add.mean():7.0f} {100 * sh.mean():5.0f}  {rms.mean():9.3f} "
                  f"{np.median(rms):7.3f}  {nr.mean():10.2f} {np.median(nr):7.2f}  {uu.mean():5.2f} "
                  f"{cc.mean():6.2f}  {100 * better:9.0f}%")
        with open(f"{out}/summary_by_grid.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["grid_size_k", "forward_calls_mean", "catalog_added_mean", "nbr_share_pct_mean",
                        "RMS_mean", "RMS_median", "nRMS_MoR_mean", "nRMS_MoR_median", "U_mean", "cov95_mean",
                        "pct_samples_RMS_better_than_original"])
            w.writerows(summary)

        cmp_recs = self._load_compare([int(r["sample_idx"]) for r in recs])
        if cmp_recs:
            print(f"\ncomparison files found for {len(cmp_recs)} of these samples ({self.compare_dir})")
            rr = np.array([r["ensiol_final_rms"] for r in cmp_recs])
            nn = np.array([r["ensiol_final_nrms_mor"] for r in cmp_recs])
            cc = np.array([r["ensiol_total_calls"] for r in cmp_recs])
            print(f"  {'EnsIOL (own stopping point)':28s} calls median {np.median(cc):6.0f} | "
                  f"RMS mean {rr.mean():.3f} | nRMS mean {nn.mean():.2f}")
            bud = cmp_recs[0]["budgets"]
            for j, B in enumerate(bud):
                if B in (1500, 20000):
                    rr = np.array([r["mcmc_catalog_rms"][j] for r in cmp_recs])
                    nn = np.array([r["mcmc_catalog_nrms_mor"][j] for r in cmp_recs])
                    print(f"  MCMC at budget {int(B):6d}        RMS mean {rr.mean():.3f} | nRMS mean {nn.mean():.2f}")

        if self.make_figures:
            self._summary_figure(recs, cmp_recs, summary, rms0, nr0)

    def _summary_figure(self, recs, cmp_recs, summary, rms0, nr0) -> None:
        """Median and interquartile range over samples, against forward calls."""
        stack = lambda key: np.array([r[key] for r in recs])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        x = np.array([s[1] for s in summary])
        n_modes = self.sweep.sweep.n_modes
        for ax, key, ylab in zip(axes, ("rms", "nrms_mor"), ("RMS error (log$_{10}$ Ω·m)", "nRMS (mean of responses)")):
            v = stack(f"grid_{key}")
            ax.fill_between(x, np.percentile(v, 25, axis=0), np.percentile(v, 75, axis=0), color="C1", alpha=0.15)
            ax.plot(x, np.median(v, axis=0), "s-", color="C1", ms=4, label=f"PC grid ({n_modes}D)")
            if cmp_recs:
                bud = cmp_recs[0]["budgets"]
                pos = bud > 0
                m = np.array([r[f"mcmc_catalog_{key}"] for r in cmp_recs])[:, pos]
                ax.fill_between(bud[pos], np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0),
                                color="C2", alpha=0.15)
                ax.plot(bud[pos], np.median(m, axis=0), "^-", color="C2", ms=4, label="MCMC (catalog)")
                e = np.array([r[f"ensiol_final_{key}"] for r in cmp_recs])
                ec = np.array([r["ensiol_total_calls"] for r in cmp_recs])
                ax.errorbar(np.median(ec), np.median(e),
                            yerr=[[np.median(e) - np.percentile(e, 25)], [np.percentile(e, 75) - np.median(e)]],
                            fmt="o", color="C0", ms=6, capsize=3, label="EnsIOL (own stopping point)")
            ax.axhline(np.median(rms0 if key == "rms" else nr0), color="0.4", ls=":", lw=1, label="Original catalog")
            ax.set_xscale("log")
            ax.set_xlabel("Forward-model calls")
            ax.set_ylabel(ylab)
            ax.legend(fontsize=8)
        axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
        fig.suptitle(f"Naive PC-grid baseline: median and interquartile range over {len(recs)} test samples",
                     fontsize=10)
        fig.savefig(f"{self.out_dir}/pc_grid_accuracy_vs_cost.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    plt.switch_backend("Agg")      # command line / cluster: no display needed
    p = argparse.ArgumentParser(description="Naive principal-component grid baseline for catalog growth.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/kilauea")
    p.add_argument("--out-dir", default="outputs/pc_sweep")
    p.add_argument("--compare-dir", default="outputs/ensiol_vs_mcmc",
                   help="output folder of run_ensiol_vs_mcmc.py (optional, adds EnsIOL and MCMC to the summary figure)")
    p.add_argument("--n-modes", type=int, default=PCGridConfig().n_modes, help="number of leading principal components")
    p.add_argument("--grid-sizes", type=int, nargs="*", default=list(PCGridConfig().grid_sizes), help="values per axis")
    p.add_argument("--samples", type=int, nargs="*", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    a = p.parse_args(argv)

    pc = PCGridConfig(n_modes=a.n_modes, grid_sizes=tuple(a.grid_sizes))
    ctx = ExperimentContext.from_data_dir(a.data_dir, with_test=True, pc_grid=pc)
    exp = PCSweepBaseline(ctx, a.out_dir, grid_sizes=a.grid_sizes, compare_dir=a.compare_dir,
                          make_figures=not a.no_figures, resume=not a.no_resume, workers=a.workers)
    items = list(range(len(ctx.test))) if a.samples is None else a.samples

    if not a.collect_only:
        exp.run(items)
        sw = ctx.pc_sweep
        if all(k in sw.grids for k in a.grid_sizes):         # the grids were built in this run
            b = sw.basis
            with open(f"{a.out_dir}/run_config.json", "w") as f:
                json.dump(dict(data_dir=a.data_dir, N_MODES=a.n_modes, GRID_SIZES=a.grid_sizes,
                               var_explained=b.var_explained, coef_min=b.cmin.tolist(), coef_max=b.cmax.tolist(),
                               grid_nominal={k: sw.grids[k].n_nominal for k in a.grid_sizes},
                               grid_forward_calls={k: sw.grids[k].n_valid for k in a.grid_sizes},
                               N_NEIGHBORS=ctx.config.n_neighbors, d_min=ctx.d_min,
                               bounds_ohm_m=[ctx.config.res_min, ctx.config.res_max],
                               ESTIMATOR="single backward EnsCGP update (no correction)",
                               sweep="regular grid over the leading principal components of the original "
                                     "catalog (natural-log resistivity), coefficient range = catalog min/max, "
                                     "no information from the observation"), f, indent=2)
    exp.collect(items)
    print(f"Done. Outputs in: {a.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
