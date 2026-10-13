#!/usr/bin/env python
"""Draw the synthetic test models for the new geologic setting (noiseless).

Writes the adaptation and held-out splits (models and noiseless responses) into
--out-dir/{adapt,holdout}/noise_0/ together with the specification used. The
observations used by the experiments (``kilauea_1d_*_test.txt``) are the
adaptation split with observation noise applied; those files ship with the
repository in data/kilauea/.
"""
import argparse
import os

import numpy as np

from ensiol.forward import MTForward1D, SerialExecutor
from ensiol.priors import NewSettingSpec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default="data/kilauea")
    p.add_argument("--out-dir", default="data/new_setting_test")
    p.add_argument("--n-adapt", type=int, default=100)
    p.add_argument("--n-holdout", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()

    depths = np.loadtxt(f"{a.grid_dir}/kilauea_depth_interfaces_m.txt")
    freqs = np.loadtxt(f"{a.grid_dir}/kilauea_1d_freqs_prior.txt")
    spec = NewSettingSpec()
    fw = MTForward1D(depths)
    a50, p50 = fw.response(np.full(depths.size - 1, 50.0), freqs)
    assert np.allclose(a50, 50.0, rtol=1e-6) and np.allclose(p50, 45.0, atol=1e-6), "half-space check failed"
    for name, rho in zip(("adapt", "holdout"), spec.sample(depths, a.n_adapt, a.n_holdout, a.seed)):
        if len(rho) == 0:
            continue
        app, ph = fw.simulate(rho, freqs, SerialExecutor())
        d = f"{a.out_dir}/{name}/noise_0"
        os.makedirs(d, exist_ok=True)
        np.savetxt(f"{d}/kilauea_1d_ResModels.txt", rho)
        np.savetxt(f"{d}/kilauea_1d_AppRes.txt", app)
        np.savetxt(f"{d}/kilauea_1d_Phase.txt", ph)
        print(f"{name}: {rho.shape}, rho range {rho.min():.3g} .. {rho.max():.3g} ohm-m")


if __name__ == "__main__":
    main()
