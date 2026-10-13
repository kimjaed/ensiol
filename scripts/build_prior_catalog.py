#!/usr/bin/env python
"""Rebuild the prior catalog (the Gabbs Valley prior transferred to Kilauea).

Reads the Gabbs Valley station ensemble and Kilauea's depth interfaces and
frequencies, samples the Gaussian prior, forward-models it, and writes
``kilauea_1d_{ResModels,AppRes,Phase}_prior.txt`` into --out-dir. With the
default settings this reproduces the catalog used in the manuscript.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ensiol.forward import MTForward1D
from ensiol.priors import GabbsValleyStatistics, GMRFPrior


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gv-ensemble", default="data/gabbs_valley/real_data_ensemble.npy")
    p.add_argument("--out-dir", default="data/kilauea")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--corr-len-cells", type=float, default=2.0)
    p.add_argument("--modes", type=int, default=30)
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()

    depths = np.loadtxt(f"{a.out_dir}/kilauea_depth_interfaces_m.txt")
    freqs = np.loadtxt(f"{a.out_dir}/kilauea_1d_freqs_prior.txt")
    stats = GabbsValleyStatistics.from_file(a.gv_ensemble, n_components=a.modes)
    print(f"Gabbs Valley: {stats.n_stations} stations, {stats.modes.shape[1]} modes "
          f"({100 * stats.var_explained:.1f}% variance)")
    prior = GMRFPrior(stats, depths, corr_len_cells=a.corr_len_cells, sigma=a.sigma)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res, app, ph = prior.build_catalog(a.n, freqs, MTForward1D(depths), ex, seed=a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    np.savetxt(f"{a.out_dir}/kilauea_1d_ResModels_prior.txt", res)
    np.savetxt(f"{a.out_dir}/kilauea_1d_AppRes_prior.txt", app)
    np.savetxt(f"{a.out_dir}/kilauea_1d_Phase_prior.txt", ph)
    print(f"wrote {res.shape[0]} members to {a.out_dir}/")


if __name__ == "__main__":
    main()
