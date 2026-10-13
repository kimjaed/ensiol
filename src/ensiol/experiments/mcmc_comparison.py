"""EnsIOL versus MCMC as ways of growing the catalog for one observation.

Both start from the same catalog, MCMC samples a posterior whose
prior is that catalog's own Gaussian, accepted MCMC states join the catalog under
the same non-redundancy rule as EnsIOL's candidates, and both grown catalogs
are scored by the same single backward EnsCGP update at forward-call budgets.
For EnsIOL the catalog at budget ``B`` is the original catalog plus its first
``B`` additions (each addition costs exactly one forward call; the surrogate
loop makes none). Beyond EnsIOL's own total, its final catalog is used.

``ensiol_budgets``:
  ``"grid"``        EnsIOL scored on the same budget grid as MCMC (the 100-sample comparison);
  ``"iterations"``  EnsIOL scored at its own cumulative call count after every iteration
                    (the per-sample figure); MCMC stays on the coarse grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ..context import ExperimentContext
from ..forward import SerialExecutor
from ..plotting import ComparisonStyle, accuracy_vs_cost_summary, plot_comparison
from .base import Experiment

DEFAULT_BUDGETS = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000,
                   5000, 7000, 10000, 15000, 20000]
METRICS = ("calls", "catalog_added", "rms", "nrms_rom", "nrms_mor")


def cost_series(rec: dict, method: str, ensiol_budgets: str = "grid"):
    """Arrays for :func:`cost_figure`: the original catalog (0 calls) first, then the points.

    In ``grid`` mode EnsIOL's budgets above its own total repeat its final catalog, so
    its curve is drawn up to the budget below its total and then its own final point.
    """
    if method == "mcmc":
        return {k: rec[f"mcmc_catalog_{k}"] for k in ("calls", "rms", "nrms_rom", "nrms_mor")}
    out = {k: np.asarray(rec[f"ensiol_{k}"]) for k in ("calls", "rms", "nrms_rom", "nrms_mor")}
    if ensiol_budgets == "iterations":
        return out
    n_iol = int(rec["ensiol_total_calls"])
    b = np.asarray(rec["budgets_ensiol"])
    keep = (b > 0) & (b < n_iol)
    fin = {"rms": rec["ensiol_final_rms"], "nrms_rom": rec["ensiol_final_nrms_rom"], "nrms_mor": rec["ensiol_final_nrms_mor"]}
    res = {"calls": np.concatenate([[0], out["calls"][keep], [n_iol]])}
    for k in ("rms", "nrms_rom", "nrms_mor"):
        res[k] = np.concatenate([[out[k][0]], out[k][keep], [fin[k]]])
    return res


class EnsIOLvsMCMC(Experiment):
    def __init__(self, ctx: ExperimentContext, out_dir: str, ensiol_dir: str = None, ensiol_budgets: str = "grid",
                 budgets: Sequence[int] = None, make_figures: bool = True, **kw):
        super().__init__(out_dir, **kw)
        assert ctx.test is not None, "context was built without a test set"
        self.ctx, self.ensiol_dir, self.mode, self.make_figures = ctx, ensiol_dir, ensiol_budgets, make_figures
        mc = ctx.pcn.max_calls
        b = [x for x in (budgets or DEFAULT_BUDGETS) if x <= mc]
        self.budgets = b if b[-1] == mc else b + [mc]
        os.makedirs(f"{out_dir}/per_sample", exist_ok=True)
        os.makedirs(f"{out_dir}/trajectory_figs", exist_ok=True)

    def item_path(self, idx) -> str:
        return f"{self.out_dir}/per_sample/sample_{idx:03d}.npz"

    def _ensiol_additions(self, idx, case, seed, base):
        """Saved EnsIOL additions from the evaluation run if available, else rerun EnsIOL."""
        f = f"{self.ensiol_dir}/sample_{idx:03d}.npz" if self.ensiol_dir else None
        if f and os.path.exists(f):
            r = np.load(f, allow_pickle=True)
            assert np.allclose(r["truth_log10"], case.truth_log10), "cached EnsIOL record does not match this sample"
            return (r["added_Labels"], r["added_AppRes_log"], r["added_Phase"], r["hist_n_accepted"], "loaded")
        run = self.ctx.augment(case.observation, seed, truth_linear=case.truth_linear)
        return (*run.additions, run.growth.n_accepted, "rerun")

    def run_item(self, idx: int) -> dict:
        t0 = time.time()
        ctx = self.ctx
        ex = SerialExecutor()
        case = ctx.test.case(idx)
        obs, truth = case.observation, case.truth_log10
        seed = ctx.config.base_seed + idx
        base, _ = ctx.starting_catalog(case.truth_linear)

        iL, iA, iP, n_acc, src = self._ensiol_additions(idx, case, seed, base)
        n_iol = iL.shape[1]
        mc = ctx.sampler.run(base, obs, seed, ex)

        if self.mode == "iterations":
            b_iol = np.unique(np.concatenate([[0], np.cumsum(n_acc)]).astype(int))
        else:
            b_iol = np.array(self.budgets, int)
        b_mc = np.array(self.budgets, int)

        score = lambda k: ctx.scorer.score_catalog(base.with_additions(iL, iA, iP, k=k), obs, truth, seed, ex)
        rows = dict(ensiol=[], mcmc_catalog=[])
        for B in b_iol:
            k = min(int(B), n_iol)
            rows["ensiol"].append(dict(calls=k, catalog_added=k, **score(k)))
        mL, mA, mP = mc.add_L, mc.add_A, mc.add_P
        for B in b_mc:
            calls_B, kB = mc.at_budget(int(B))
            s = ctx.scorer.score_catalog(base.with_additions(mL, mA, mP, k=kB), obs, truth, seed, ex)
            rows["mcmc_catalog"].append(dict(calls=calls_B, catalog_added=kB, **s))
        fin = score(n_iol)

        out = dict(sample_idx=idx, seed=seed, truth_log10=truth, ensiol_source=src, ensiol_total_calls=n_iol,
                   budgets_ensiol=b_iol, budgets_mcmc=b_mc, budgets=b_mc,
                   mcmc_acc=mc.acc, mcmc_beta=mc.beta, mcmc_total_calls=mc.total_calls,
                   mcmc_n_added_total=mc.total_added, ensiol_final_rms=fin["rms"],
                   ensiol_final_nrms_rom=fin["nrms_rom"], ensiol_final_nrms_mor=fin["nrms_mor"])
        for m, lst in rows.items():
            for key in METRICS:
                out[f"{m}_{key}"] = np.array([r[key] for r in lst])
            out[f"{m}_mean_log10"] = np.array([r["mean_log10"] for r in lst])
        out["runtime_s"] = time.time() - t0
        if self.make_figures:
            nk = "nrms_" + ctx.scoring.nrms_convention.lower()
            diag = ComparisonStyle(fig_width=12, panel_height=4.8, fs_label=10, fs_tick=9, fs_legend=9, ms=4,
                                   titles=True, annotate_done=True, annotate_kept=True)
            fig = plot_comparison(cost_series(out, "ensiol", self.mode), cost_series(out, "mcmc"), n_iol,
                                  mc.total_added, mc.total_calls, diag, nrms_key=nk)
            fig.suptitle(f"Sample {idx}")
            fig.savefig(f"{self.out_dir}/trajectory_figs/accuracy_vs_cost_sample_{idx:03d}.png", dpi=150)
            plt.close(fig)
        return out

    def summary_line(self, r: dict) -> str:
        return (f"sample {int(r['sample_idx']):3d}  EnsIOL {int(r['ensiol_total_calls'])} calls ({r['ensiol_source']})  "
                f"RMS at max budget: EnsIOL {r['ensiol_rms'][-1]:.3f} | MCMC catalog {r['mcmc_catalog_rms'][-1]:.3f}  "
                f"MCMC added {int(r['mcmc_n_added_total'])}  ({r['runtime_s']:.0f} s)")

    def collect(self, items: Sequence[int]) -> None:
        recs = [dict(np.load(self.item_path(i), allow_pickle=True)) for i in items if os.path.exists(self.item_path(i))]
        if not recs:
            print("no finished samples to collect", flush=True)
            return
        print(f"collected {len(recs)} samples", flush=True)
        methods = [("ensiol", "EnsIOL (catalog)", "C0", "budgets_ensiol"), ("mcmc_catalog", "MCMC (catalog)", "C2", "budgets_mcmc")]
        with open(f"{self.out_dir}/results_long.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "method", "budget", "forward_calls", "catalog_added", "RMS_log10", "nRMS_RoM", "nRMS_MoR"])
            for r in recs:
                for m, _, _, bk in methods:
                    for j, B in enumerate(r[bk]):
                        w.writerow([int(r["sample_idx"]), m, int(B), int(r[f"{m}_calls"][j]), int(r[f"{m}_catalog_added"][j]),
                                    r[f"{m}_rms"][j], r[f"{m}_nrms_rom"][j], r[f"{m}_nrms_mor"][j]])
        if not all(np.array_equal(r["budgets_ensiol"], r["budgets_mcmc"]) for r in recs):
            print("per-method budget grids differ (iterations mode): skipping the pooled summary")
            return
        budgets = recs[0]["budgets_mcmc"]
        stack = lambda key: np.array([r[key] for r in recs])
        tot = stack("ensiol_total_calls")
        print(f"\nEnsIOL total forward calls: median {np.median(tot):.0f} [IQR {np.percentile(tot, 25):.0f}-{np.percentile(tot, 75):.0f}]")
        summary = []
        for j, B in enumerate(budgets):
            for m, lab, _, _ in methods:
                rms, nr = stack(f"{m}_rms")[:, j], stack(f"{m}_nrms_mor")[:, j]
                nr2 = stack(f"{m}_nrms_rom")[:, j]
                summary.append([int(B), m, np.nanmean(rms), np.nanmedian(rms), np.nanmean(nr), np.nanmedian(nr),
                                np.nanmean(nr2), np.nanmedian(nr2)])
        with open(f"{self.out_dir}/summary_by_budget.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["budget", "method", "RMS_mean", "RMS_median", "nRMS_MoR_mean", "nRMS_MoR_median", "nRMS_RoM_mean", "nRMS_RoM_median"])
            w.writerows(summary)
        nk = "nrms_" + self.ctx.scoring.nrms_convention.lower()
        series = {m: {k: stack(f"{m}_{k}") for k in ("rms", "nrms_rom", "nrms_mor")} for m, *_ in methods}
        fig = accuracy_vs_cost_summary(budgets, series, {m: l for m, l, _, _ in methods}, {m: c for m, _, c, _ in methods},
                                       float(np.median(tot)), len(recs), nrms_key=nk)
        fig.savefig(f"{self.out_dir}/accuracy_vs_cost.png", dpi=150); plt.close(fig)


def main(argv=None):
    plt.switch_backend("Agg")      # command line / cluster: no display needed
    p = argparse.ArgumentParser(description="EnsIOL vs. MCMC catalog growth.")
    p.add_argument("--data-dir", default="data/kilauea")
    p.add_argument("--out-dir", default="outputs/ensiol_vs_mcmc")
    p.add_argument("--ensiol-dir", default=None, help="per_sample folder of the synthetic run (reused if present)")
    p.add_argument("--ensiol-budgets", choices=("grid", "iterations"), default="grid")
    p.add_argument("--samples", type=int, nargs="*", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--mcmc-max-calls", type=int, default=20000)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    a = p.parse_args(argv)

    from ..baselines import PCNConfig
    ctx = ExperimentContext.from_data_dir(a.data_dir, with_test=True, pcn=PCNConfig(max_calls=a.mcmc_max_calls))
    exp = EnsIOLvsMCMC(ctx, a.out_dir, ensiol_dir=a.ensiol_dir, ensiol_budgets=a.ensiol_budgets,
                       make_figures=not a.no_figures, resume=not a.no_resume, workers=a.workers)
    items = list(range(len(ctx.test))) if a.samples is None else a.samples
    with open(f"{a.out_dir}/run_config.json", "w") as f:
        json.dump(dict(data_dir=a.data_dir, ensiol_dir=a.ensiol_dir, ensiol_budgets=a.ensiol_budgets, pcn=ctx.pcn.to_dict(),
                       budgets=exp.budgets, config=ctx.config.to_dict(), d_min=ctx.d_min,
                       ESTIMATOR="single backward EnsCGP update (no correction)",
                       prior="Gaussian, catalog mean/cov, truncated to [5e-3, 5e5] ohm-m", sampler="pCN"), f, indent=2)
    if not a.collect_only:
        exp.run(items)
    exp.collect(items)
    print(f"Done. Outputs in: {a.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
