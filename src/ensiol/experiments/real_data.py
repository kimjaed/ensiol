"""EnsIOL on real stations. No truth exists, so only data misfit is reported."""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ..config import LN10
from ..context import ExperimentContext, station_seed
from ..data import RealStation
from ..enscgp import Estimate
from ..plotting import plot_evolution
from ..snapshots import SnapshotEvaluator
from .base import Experiment


def select_stations(real_dir: str, pattern: str = "*_1d_response.dat", min_freqs: int = 10,
                    only: Sequence[str] = None) -> List[tuple]:
    """(station_id, path) pairs with at least ``min_freqs`` frequencies.

    Stations with fewer frequencies constrain the resistivity structure weakly
    and give highly uncertain models, so they are skipped.
    """
    found = RealStation.discover(real_dir, pattern)
    if only is not None:
        found = [s for s in found if s[0] in set(only)]
    keep, skipped = [], []
    for sid, path in found:
        n = RealStation.load(sid, path).observation.n_freqs
        (keep if n >= min_freqs else skipped).append((sid, path))
    print(f"{len(found)} station files found; {len(keep)} with >= {min_freqs} frequencies kept, "
          f"{len(skipped)} skipped", flush=True)
    return keep


class RealDataApplication(Experiment):
    """Each station runs independently from a fresh copy of the prior catalog,
    on its own frequency grid, exactly as each synthetic test sample does."""

    def __init__(self, ctx: ExperimentContext, out_dir: str, stations: List[tuple], n_rows: int = 5,
                 make_figures: bool = True, **kw):
        super().__init__(out_dir, **kw)
        self.ctx, self.stations, self.n_rows, self.make_figures = ctx, dict(stations), n_rows, make_figures
        self.snapshots = SnapshotEvaluator(ctx.enscgp, n_rows)
        os.makedirs(f"{out_dir}/per_station", exist_ok=True)
        os.makedirs(f"{out_dir}/evolution_figs", exist_ok=True)

    def item_path(self, sid) -> str:
        return f"{self.out_dir}/per_station/{sid}.npz"

    def run_item(self, sid: str) -> dict:
        ctx = self.ctx
        st = RealStation.load(sid, self.stations[sid])
        obs = st.observation
        seed = station_seed(ctx.config.base_seed, sid)
        cat_f = ctx.grid.freqs
        if obs.freqs.min() < cat_f.min() or obs.freqs.max() > cat_f.max():
            print(f"  WARNING [{sid}]: station band [{obs.freqs.min():.4g}, {obs.freqs.max():.4g}] Hz extends "
                  f"beyond the catalog band [{cat_f.min():.4g}, {cat_f.max():.4g}] Hz; those frequencies are "
                  "extrapolated, not interpolated.", flush=True)
        run = ctx.augment(obs, seed)
        iters = self.snapshots.row_iterations(run.growth)
        rows = run.evolution_rows(iterations=iters, n_rows=self.n_rows)
        assert rows[-1]["catalog_size"] == run.catalog.n
        base = ctx.scorer.summarize(Estimate.from_dict(rows[0]), obs)
        aug = ctx.scorer.summarize(Estimate.from_dict(rows[-1]), obs)
        if self.make_figures:
            fig = plot_evolution(rows, obs, ctx.grid.depths, ctx.whitener.ref_std, ctx.config.noise)
            fig.savefig(f"{self.out_dir}/evolution_figs/fig_evolution_{sid}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
        addL, addA, addP = run.additions
        out = dict(station_id=sid, filepath=str(st.path), seed=seed, n_freqs=obs.n_freqs,
                   freq_min=float(obs.freqs.min()), freq_max=float(obs.freqs.max()),
                   n_iterations=len(run.growth.history), exit_reason=run.growth.exit_reason,
                   n_added=run.n_added, runtime_s=run.runtime_s, real_freqs=obs.freqs,
                   obs_appres_log10=obs.app_res_log / LN10, obs_phase=obs.phase,
                   added_Labels=addL, added_AppRes_log=addA, added_Phase=addP,
                   hist_nRMS_approx=np.array([h.nRMS_approx for h in run.growth.history]),
                   hist_n_accepted=run.growth.n_accepted, evo_iterations=np.array(iters),
                   evo_catalog_size=np.array([r["catalog_size"] for r in rows]),
                   evo_nRMS_MoR=np.array([r["nRMS"] for r in rows]),
                   base_ens_Labels=rows[0]["Labels_est"], base_ens_AppRes_log=rows[0]["AppRes_real"],
                   base_ens_Phase=rows[0]["Phase_real"], aug_ens_Labels=rows[-1]["Labels_est"],
                   aug_ens_AppRes_log=rows[-1]["AppRes_real"], aug_ens_Phase=rows[-1]["Phase_real"])
        for arm, d in (("base", base), ("aug", aug)):
            for k, v in d.items():
                out[f"{arm}_{k}"] = v
        return out

    def summary_line(self, r: dict) -> str:
        return (f"{str(r['station_id']):12s}  {int(r['n_freqs']):2d} freqs  "
                f"nRMS(MoR) {r['base_nRMS_MoR']:.3f} -> {r['aug_nRMS_MoR']:.3f}  added={r['n_added']:4d}  "
                f"iters={r['n_iterations']:3d}  exit={r['exit_reason']}  ({r['runtime_s']:.0f} s)")

    def collect(self, items: Sequence[str]) -> None:
        files = [(s, self.item_path(s)) for s in items if os.path.exists(self.item_path(s))]
        if not files:
            print("no finished stations to collect", flush=True)
            return
        recs = [(s, dict(np.load(f, allow_pickle=True))) for s, f in files]
        print(f"collected {len(recs)} stations", flush=True)
        fields = ["station_id", "n_freqs", "freq_min", "freq_max", "n_iterations", "exit_reason", "n_added",
                  "runtime_s", "base_nRMS_MoR", "aug_nRMS_MoR", "base_nRMS_RoM", "aug_nRMS_RoM"]
        with open(f"{self.out_dir}/station_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for _, r in recs:
                w.writerow([r[k].item() if hasattr(r[k], "item") else r[k] for k in fields])
        g = lambda k: np.array([r[k] for _, r in recs])
        bm, am, br, ar = g("base_nRMS_MoR"), g("aug_nRMS_MoR"), g("base_nRMS_RoM"), g("aug_nRMS_RoM")
        print(f"nRMS, mean of responses:  median {np.median(bm):.3f} -> {np.median(am):.3f} | improved in {np.mean(am < bm) * 100:.0f}% of stations")
        print(f"nRMS, response of mean:   median {np.median(br):.3f} -> {np.median(ar):.3f} | improved in {np.mean(ar < br) * 100:.0f}% of stations")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, a, b, lab in [(axes[0], bm, am, "nRMS, mean of responses"), (axes[1], br, ar, "nRMS, response of mean")]:
            ax.scatter(a, b, s=20, alpha=0.7)
            lim = [0, 1.05 * max(a.max(), b.max())]
            ax.plot(lim, lim, "k--", lw=1); ax.set_xlabel(f"{lab}, baseline"); ax.set_ylabel(f"{lab}, augmented")
        fig.tight_layout(); fig.savefig(f"{self.out_dir}/baseline_vs_augmented.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    plt.switch_backend("Agg")      # command line / cluster: no display needed
    p = argparse.ArgumentParser(description="EnsIOL on real Kilauea stations.")
    p.add_argument("--data-dir", default="data/kilauea", help="prior catalog, frequency grid and depth interfaces")
    p.add_argument("--real-dir", default="data/kilauea_stations", help="directory of *_1d_response.dat files")
    p.add_argument("--out-dir", default="outputs/real_data")
    p.add_argument("--pattern", default="*_1d_response.dat")
    p.add_argument("--min-freqs", type=int, default=10, help="skip stations with fewer frequencies")
    p.add_argument("--stations", nargs="*", default=None, help="station ids to run (default: all kept)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    a = p.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    ctx = ExperimentContext.from_data_dir(a.data_dir, with_test=False)
    stations = select_stations(a.real_dir, a.pattern, a.min_freqs, a.stations)
    exp = RealDataApplication(ctx, a.out_dir, stations, make_figures=not a.no_figures,
                              resume=not a.no_resume, workers=a.workers)
    ids = [s for s, _ in stations]
    with open(f"{a.out_dir}/run_config.json", "w") as f:
        json.dump(dict(data_dir=a.data_dir, real_dir=a.real_dir, config=ctx.config.to_dict(), d_min=ctx.d_min,
                       ESTIMATOR="single backward EnsCGP update (no correction)", min_freqs=a.min_freqs,
                       n_prior=len(ctx.prior), n_stations=len(ids), station_ids=ids), f, indent=2)
    if not a.collect_only:
        exp.run(ids)
    exp.collect(ids)
    print(f"Done. Outputs in: {a.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
