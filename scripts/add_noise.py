#!/usr/bin/env python
"""Add observation noise to noiseless synthetic MT responses.

Reads clean models and responses (as written by build_test_set.py), perturbs the apparent
resistivity and phase with relative-error noise, and writes the noisy data set. The models
are copied unchanged.

Noise model (the one used for the manuscript's test set):
  rel_err               assumed relative error in the MT transfer functions (default 0.1 = 10%)
  apparent resistivity  rho_noisy = rho * max(1 + 2*rel_err * N(0, 1), 1e-8)   (multiplicative)
  phase                 phi_noisy = phi + degrees(arctan(rel_err)) * N(0, 1)   (additive, degrees)

Input files in --in-dir:  kilauea_1d_{ResModels,AppRes,Phase}.txt and, optionally,
kilauea_1d_freqs.txt (otherwise --freqs is used).

Output files in --out-dir (default: a sibling of --in-dir named noise_<percent>):
  kilauea_1d_{ResModels,AppRes,Phase,freqs}.txt, noise_info.json, and with --plot noisy_data_sample.png.
With --as-test-set the files are named kilauea_1d_{ResModels,AppRes,Phase,freqs}_test.txt, the names
the experiments read.

Example:
  python scripts/add_noise.py --in-dir data/new_setting_test/adapt/noise_0 --rel-err 0.1 --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ensiol.config import NoiseModel

PREFIX = "kilauea_1d_"


def add_observation_noise(app_res, phase, noise: NoiseModel, rng):
    """Noisy copies of linear apparent resistivity (ohm-m) and phase (degrees).

    Apparent resistivity is drawn first and phase second, from one random stream.
    """
    factor = np.clip(1.0 + noise.app_res_sigma * rng.standard_normal(app_res.shape), 1e-8, None)
    app_res_noisy = app_res * factor
    phase_noisy = phase + noise.phase_sigma * rng.standard_normal(phase.shape)
    return app_res_noisy, phase_noisy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="data/new_setting_test/adapt/noise_0", help="folder with the clean data")
    ap.add_argument("--out-dir", default=None, help="output folder (default: sibling of --in-dir, noise_<percent>)")
    ap.add_argument("--rel-err", type=float, default=0.1, help="relative error of the transfer functions")
    ap.add_argument("--seed", type=int, default=0, help="random seed (recorded in noise_info.json)")
    ap.add_argument("--freqs", default="data/kilauea/kilauea_1d_freqs_prior.txt",
                    help="frequency file, used when --in-dir has none")
    ap.add_argument("--as-test-set", action="store_true", help="name the outputs kilauea_1d_*_test.txt")
    ap.add_argument("--plot", action="store_true", help="save a clean-versus-noisy plot of the first sample")
    a = ap.parse_args()

    in_dir = Path(a.in_dir)
    out_dir = Path(a.out_dir) if a.out_dir else in_dir.parent / f"noise_{int(round(100 * a.rel_err))}"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = np.atleast_2d(np.loadtxt(in_dir / f"{PREFIX}ResModels.txt"))
    app_res = np.atleast_2d(np.loadtxt(in_dir / f"{PREFIX}AppRes.txt"))
    phase = np.atleast_2d(np.loadtxt(in_dir / f"{PREFIX}Phase.txt"))
    freqs_path = in_dir / f"{PREFIX}freqs.txt"
    freqs = np.atleast_1d(np.loadtxt(freqs_path if freqs_path.exists() else a.freqs))

    assert app_res.shape == phase.shape, "AppRes and Phase must have the same shape"
    assert app_res.shape[0] == res.shape[0], "number of models and responses differ"
    assert app_res.shape[1] == freqs.size, "number of frequencies does not match the responses"
    assert np.all(app_res > 0), "apparent resistivity must be positive"

    noise = NoiseModel(rel_err=a.rel_err)
    rng = np.random.default_rng(a.seed)
    app_res_noisy, phase_noisy = add_observation_noise(app_res, phase, noise, rng)

    suffix = "_test" if a.as_test_set else ""
    np.savetxt(out_dir / f"{PREFIX}ResModels{suffix}.txt", res)
    np.savetxt(out_dir / f"{PREFIX}AppRes{suffix}.txt", app_res_noisy)
    np.savetxt(out_dir / f"{PREFIX}Phase{suffix}.txt", phase_noisy)
    np.savetxt(out_dir / f"{PREFIX}freqs{suffix}.txt", freqs)
    with open(out_dir / "noise_info.json", "w") as f:
        json.dump(dict(rel_err=a.rel_err, app_res_noise_level=noise.app_res_sigma,
                       phase_noise_level_deg=noise.phase_sigma, seed=a.seed, n_samples=int(res.shape[0]),
                       source=str(in_dir)), f, indent=2)

    print(f"rel_err {a.rel_err}: app_res_noise_level {noise.app_res_sigma}, "
          f"phase_noise_level {noise.phase_sigma:.4f} deg, seed {a.seed}")
    print(f"saved {res.shape[0]} noisy samples to {out_dir.resolve()}")

    if a.plot:
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")
        periods = 1.0 / freqs
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for ax, clean, noisy, label in ((axes[0], app_res, app_res_noisy, "Apparent resistivity (ohm-m)"),
                                        (axes[1], phase, phase_noisy, "Phase (degrees)")):
            ax.plot(periods, clean[0], "o--", label="Clean")
            ax.plot(periods, noisy[0], "o--", label="Noisy")
            ax.set_xscale("log")
            ax.set_ylabel(label)
            ax.legend()
        axes[1].set_xlabel("Period (s)")
        fig.suptitle("First sample")
        fig.savefig(out_dir / "noisy_data_sample.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()