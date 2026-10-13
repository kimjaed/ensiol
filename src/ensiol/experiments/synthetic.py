"""EnsIOL on a set of synthetic test cases (the manuscript's Table I)."""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from ..config import LN10, EnsIOLConfig, ScoringConfig
from ..context import ExperimentContext
from ..enscgp import Estimate
from ..metrics import coverage_tables, rms_log10
from ..plotting import plot_evolution
from ..snapshots import SnapshotEvaluator
from .base import Experiment


class SyntheticEvaluation(Experiment):
    """Each test sample is independent: EnsIOL starts from a fresh copy of the
    prior catalog (members identical to the sample's truth removed), and the
    catalog is scored by a single backward EnsCGP update before and after."""

    def __init__(self, ctx: ExperimentContext, out_dir: str, n_rows: int = 5, make_figures: bool = True, **kw):
        super().__init__(out_dir, **kw)
        assert ctx.test is not None, "context was built without a test set"
        self.ctx, self.n_rows, self.make_figures = ctx, n_rows, make_figures
        self.snapshots = SnapshotEvaluator(ctx.enscgp, n_rows)
        os.makedirs(f"{out_dir}/per_sample", exist_ok=True)
        os.makedirs(f"{out_dir}/evolution_figs", exist_ok=True)

    def item_path(self, idx) -> str:
        return f"{self.out_dir}/per_sample/sample_{idx:03d}.npz"

    def run_item(self, idx: int) -> dict:
        ctx = self.ctx
        case = ctx.test.case(idx)
        obs, truth_log10 = case.observation, case.truth_log10
        seed = ctx.config.base_seed + idx
        run = ctx.augment(obs, seed, truth_linear=case.truth_linear)
        iters = self.snapshots.row_iterations(run.growth)
        rows = run.evolution_rows(iterations=iters, n_rows=self.n_rows)
        assert rows[-1]["catalog_size"] == run.catalog.n

        base = ctx.scorer.summarize(Estimate.from_dict(rows[0]), obs, truth_log10)
        aug = ctx.scorer.summarize(Estimate.from_dict(rows[-1]), obs, truth_log10)
        if self.make_figures:
            fig = plot_evolution(rows, obs, ctx.grid.depths, ctx.whitener.ref_std, ctx.config.noise,
                                 truth_linear=case.truth_linear, exclude_halfspace=ctx.scoring.exclude_halfspace)
            fig.savefig(f"{self.out_dir}/evolution_figs/fig_synthetic_evolution_sample_{idx:03d}.png",
                        dpi=160, bbox_inches="tight")
            plt.close(fig)

        A_true, P_true = ctx.forward.response(case.truth_linear, obs.freqs)
        addL, addA, addP = run.additions
        out = dict(sample_idx=idx, seed=seed, n_leak_removed=run.n_leak_removed,
                   n_iterations=len(run.growth.history), exit_reason=run.growth.exit_reason,
                   n_added=run.n_added, runtime_s=run.runtime_s, truth_log10=truth_log10,
                   obs_appres_log10=obs.app_res_log / LN10, obs_phase=obs.phase,
                   true_appres_log10=np.log10(A_true), true_phase=P_true,
                   added_Labels=addL, added_AppRes_log=addA, added_Phase=addP,
                   hist_nRMS_approx=np.array([h.nRMS_approx for h in run.growth.history]),
                   hist_n_accepted=run.growth.n_accepted,
                   evo_iterations=np.array(iters),
                   evo_catalog_size=np.array([r["catalog_size"] for r in rows]),
                   evo_RMS_log10=np.array([rms_log10(r["Labels_est"].mean(axis=1) / LN10, truth_log10,
                                                     ctx.scoring.exclude_halfspace) for r in rows]),
                   evo_nRMS_MoR=np.array([r["nRMS"] for r in rows]))
        for arm, d in (("base", base), ("aug", aug)):
            for k, v in d.items():
                out[f"{arm}_{k}"] = v
        return out

    def summary_line(self, r: dict) -> str:
        return (f"sample {int(r['sample_idx']):3d}  RMS {r['base_RMS_log10']:.3f} -> {r['aug_RMS_log10']:.3f}  "
                f"nRMS(MoR) {r['base_nRMS_MoR']:.3f} -> {r['aug_nRMS_MoR']:.3f}  added={r['n_added']:4d}  "
                f"iters={r['n_iterations']:3d}  exit={r['exit_reason']}  ({r['runtime_s']:.0f} s)")

    # ---- aggregation -------------------------------------------------------------
    def collect(self, items: Sequence[int]) -> None:
        ctx, out = self.ctx, self.out_dir
        files = [self.item_path(i) for i in items if os.path.exists(self.item_path(i))]
        if not files:
            print("no finished samples to collect", flush=True)
            return
        recs = [dict(np.load(f, allow_pickle=True)) for f in files]
        print(f"collected {len(recs)} samples")
        scalars = ["sample_idx", "seed", "n_leak_removed", "n_iterations", "exit_reason", "n_added", "runtime_s",
                   "base_RMS_log10", "aug_RMS_log10", "base_RMS_log10_all", "aug_RMS_log10_all",
                   "base_nRMS_MoR", "aug_nRMS_MoR", "base_nRMS_RoM", "aug_nRMS_RoM"]
        item = lambda v: v.item() if hasattr(v, "item") else v
        with open(f"{out}/metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(scalars)
            for r in recs:
                w.writerow([item(r[k]) for k in scalars])
        array_keys = [k for k in recs[0] if k not in scalars and not k.startswith(("added_", "hist_", "evo_"))]
        S = {k: np.stack([r[k] for r in recs]) for k in array_keys}
        for k in scalars:
            S[k] = np.array([item(r[k]) for r in recs])
        S.update(depth_interfaces_m=ctx.grid.depths, depth_centers_m=ctx.grid.depth_centers,
                 freqs_hz=ctx.test.freqs, percentiles=np.array(ctx.scoring.percentiles))
        np.savez(f"{out}/all_samples.npz", **S)

        with open(f"{out}/evolution_rows.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "iteration", "catalog_size", "n_added", "RMS_log10", "nRMS_MoR"])
            for r in recs:
                for it, cs, rms, nr in zip(r["evo_iterations"], r["evo_catalog_size"], r["evo_RMS_log10"], r["evo_nRMS_MoR"]):
                    w.writerow([int(r["sample_idx"]), int(it), int(cs), int(cs - r["evo_catalog_size"][0]), rms, nr])

        for name, a, b in [("RMS (log10 resistivity)", S["base_RMS_log10"], S["aug_RMS_log10"]),
                           ("nRMS, mean of responses", S["base_nRMS_MoR"], S["aug_nRMS_MoR"]),
                           ("nRMS, response of mean", S["base_nRMS_RoM"], S["aug_nRMS_RoM"])]:
            print(f"{name:26s} median {np.median(a):.3f} -> {np.median(b):.3f} | mean {a.mean():.3f} -> "
                  f"{b.mean():.3f} | improved in {np.mean(b < a) * 100:.0f}% of samples")

        calib, rows_csv = coverage_tables(S, ctx.scoring.percentiles)
        calib["depth_centers_m"] = ctx.grid.depth_centers
        np.savez(f"{out}/calibration.npz", **calib)
        with open(f"{out}/calibration_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "nominal_pct", "interval", "coverage_all_layers", "coverage_excl_halfspace"])
            w.writerows(rows_csv)
        for r in rows_csv:
            print(f"{r[0]:4s}  {r[1]}%  {r[2]:8s}  coverage {100 * r[3]:5.1f}%  (excl. half-space {100 * r[4]:5.1f}%)")
        self._summary_figures(S, calib)

    def _summary_figures(self, S, calib):
        out, dc = self.out_dir, self.ctx.grid.depth_centers
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
        for ax, lvl in zip(axes[:2], (90, 95)):
            for arm, c in (("base", "0.4"), ("aug", "tab:blue")):
                ax.plot(100 * calib[f"{arm}_cov{lvl}_quantile_by_layer"], dc, "-", color=c, label=f"{arm}, quantile")
                ax.plot(100 * calib[f"{arm}_cov{lvl}_gauss_by_layer"], dc, "--", color=c, label=f"{arm}, Gaussian")
            ax.axvline(lvl, color="r", lw=1); ax.set_xlabel(f"Coverage of {lvl}% interval (%)"); ax.set_title(f"{lvl}% interval")
        axes[0].set_ylabel("Depth (m)"); axes[0].invert_yaxis(); axes[0].legend(fontsize=8)
        for arm, c in (("base", "0.4"), ("aug", "tab:blue")):
            axes[2].plot(S[f"{arm}_res_std_log10"].mean(axis=0), dc, color=c, label=arm)
        axes[2].set_xlabel("Mean ensemble std (log10 Ω·m)"); axes[2].set_title("Spread vs. depth"); axes[2].legend()
        fig.tight_layout(); fig.savefig(f"{out}/calibration_by_depth.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        for ax, arm in zip(axes, ("base", "aug")):
            ax.hist(S[f"{arm}_pit"][:, :-1].ravel(), bins=20, range=(0, 1), density=True, color="0.6")
            ax.axhline(1.0, color="r", lw=1); ax.set_xlabel("PIT")
            ax.set_title(f"PIT, {arm} (all samples and layers, excl. half-space)")
        fig.tight_layout(); fig.savefig(f"{out}/pit_histograms.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, key, lab in zip(axes, ("RMS_log10", "nRMS_MoR", "nRMS_RoM"),
                                ("RMS (log10 resistivity)", "nRMS, mean of responses", "nRMS, response of mean")):
            a, b = S[f"base_{key}"], S[f"aug_{key}"]
            ax.scatter(a, b, s=15, alpha=0.7)
            lim = [0, 1.05 * max(a.max(), b.max())]
            ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel(f"{lab}, baseline"); ax.set_ylabel(f"{lab}, augmented")
        fig.tight_layout(); fig.savefig(f"{out}/baseline_vs_augmented.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    plt.switch_backend("Agg")      # command line / cluster: no display needed
    p = argparse.ArgumentParser(description="EnsIOL on the synthetic test set (Table I).")
    p.add_argument("--data-dir", default="data/kilauea")
    p.add_argument("--out-dir", default="outputs/synthetic")
    p.add_argument("--samples", type=int, nargs="*", default=None, help="test-sample indices (default: all)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    a = p.parse_args(argv)

    ctx = ExperimentContext.from_data_dir(a.data_dir, with_test=True)
    exp = SyntheticEvaluation(ctx, a.out_dir, make_figures=not a.no_figures, resume=not a.no_resume, workers=a.workers)
    items = list(range(len(ctx.test))) if a.samples is None else a.samples
    with open(f"{a.out_dir}/run_config.json", "w") as f:
        json.dump(dict(data_dir=a.data_dir, config=ctx.config.to_dict(), d_min=ctx.d_min,
                       ESTIMATOR="single backward EnsCGP update (no correction)",
                       exclude_halfspace=ctx.scoring.exclude_halfspace, n_prior=len(ctx.prior),
                       n_test=len(ctx.test), percentiles=ctx.scoring.percentiles), f, indent=2)
    if not a.collect_only:
        exp.run(items)
    exp.collect(items)
    print(f"Done. Outputs in: {a.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
