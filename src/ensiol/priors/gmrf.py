"""The prior catalog: a Gabbs Valley-derived Gaussian prior transferred to Kilauea.

The Gabbs Valley real-station inversions give a per-depth median profile and a
low-rank covariance (SVD modes). Both are interpolated onto Kilauea's depth
grid (linear in log10 depth, clamped to the boundary values outside Gabbs
Valley's depth support), the modes are re-orthonormalised by QR, profiles are
drawn from the resulting Gaussian and smoothed by a local averaging operator,
and the catalog is forward-modelled on Kilauea's frequencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


def smoothing_operator(n_cells: int, corr_len_cells: float) -> np.ndarray:
    """Row-normalised squared-exponential kernel in grid-index distance.

    A local weighted-averaging operator, not an additive perturbation:
    applying it replaces each cell by a weighted average of its neighbours,
    which reduces roughness already present in a profile.
    """
    idx = np.arange(n_cells)
    K = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / corr_len_cells) ** 2)
    return K / K.sum(axis=1, keepdims=True)


def interpolate_depth_profile(depth_source, profile_source, depth_target, verbose_label: Optional[str] = None):
    """Linear in log10 depth; targets outside the source support take the boundary value."""
    ls, lt = np.log10(depth_source), np.log10(depth_target)
    below, above = lt < ls.min(), lt > ls.max()
    if verbose_label and (below.any() or above.any()):
        print(f"  [{verbose_label}] {below.sum()} target depth(s) shallower and {above.sum()} deeper than the "
              f"source support ({depth_source.min():.1f}..{depth_source.max():.1f} m): clamped, not extrapolated.")
    f = interp1d(ls, profile_source, kind="linear", bounds_error=False,
                 fill_value=(profile_source[0], profile_source[-1]))
    return f(lt)


@dataclass
class GabbsValleyStatistics:
    """Median profile and SVD modes of the Gabbs Valley station inversions (natural-log resistivity)."""

    depth: np.ndarray                 # (M_gv,) cell-centre depths, m
    centering_profile: np.ndarray     # (M_gv,) per-depth median of log resistivity
    modes: np.ndarray                 # (M_gv, r) orthonormal on the Gabbs Valley grid
    mode_std: np.ndarray              # (r,)
    n_stations: int
    station_names: list
    var_explained: float

    @classmethod
    def from_file(cls, path: str, n_components: int = 30) -> "GabbsValleyStatistics":
        batch = np.load(path, allow_pickle=True).item()
        rho, depth = batch["rho_matrix"], batch["zc"]
        assert np.all(np.diff(depth) > 0), "Gabbs Valley depth array (zc) must be strictly increasing"
        N = rho.shape[0]
        log_rho = np.log(rho)
        center = np.median(log_rho, axis=0)
        U, S, _ = np.linalg.svd((log_rho - center[None, :]).T, full_matrices=False)
        r = min(n_components, S.size)
        cum = np.cumsum(S ** 2 / np.sum(S ** 2))
        return cls(depth, center, U[:, :r], (S / np.sqrt(max(N - 1, 1)))[:r], N,
                   list(batch["station_names"]), float(cum[r - 1]))


class GMRFPrior:
    """Gaussian prior on Kilauea's grid built from :class:`GabbsValleyStatistics`."""

    def __init__(self, stats: GabbsValleyStatistics, depth_interfaces, corr_len_cells: float = 2.0,
                 sigma: float = 1.0, verbose: bool = True):
        self.stats, self.depth_interfaces = stats, np.asarray(depth_interfaces, float)
        assert np.all(np.diff(self.depth_interfaces) > 0), "depth interfaces must be strictly increasing"
        self.depth = 0.5 * (self.depth_interfaces[:-1] + self.depth_interfaces[1:])
        self.sigma = sigma
        r = stats.modes.shape[1]
        lab = lambda s: s if verbose else None
        self.center = interpolate_depth_profile(stats.depth, stats.centering_profile, self.depth, lab("centering_profile"))
        raw = np.zeros((self.depth.size, r))
        for j in range(r):
            raw[:, j] = interpolate_depth_profile(stats.depth, stats.modes[:, j], self.depth,
                                                  lab(f"modes 1..{r}") if j == 0 else None)
        self.modes, _ = np.linalg.qr(raw)       # orthonormal on Kilauea's grid
        self.smoother = smoothing_operator(self.depth.size, corr_len_cells)

    def _draw(self, n: int, rng) -> np.ndarray:
        x = rng.normal(0.0, self.sigma, size=(n, self.modes.shape[1]))
        log_raw = self.center[None, :] + x @ (self.modes * self.stats.mode_std[None, :]).T
        return np.exp(log_raw @ self.smoother.T)

    def sample(self, n: int, seed: int = 10, res_bounds=(5e-3, 5e5), max_rounds: int = 50) -> np.ndarray:
        """``n`` profiles (ohm-m). Whole profiles outside the bounds are redrawn."""
        rng = np.random.default_rng(seed=seed)
        accepted, needed, rejected, rounds = [], n, 0, 0
        while needed > 0 and rounds < max_rounds:
            cand = self._draw(needed, rng)
            ok = np.all((cand >= res_bounds[0]) & (cand <= res_bounds[1]), axis=1)
            accepted.append(cand[ok]); rejected += int((~ok).sum()); needed -= int(ok.sum()); rounds += 1
        if needed > 0:
            print(f"warning: {needed} sample(s) still out of bounds after {rounds} rounds; included unmodified")
            accepted.append(self._draw(needed, rng))
        print(f"redrew {rejected} out-of-bounds profile(s) over {rounds} round(s)")
        return np.vstack(accepted)[:n]

    def build_catalog(self, n: int, freqs, forward, executor, seed: int = 10):
        """Sample ``n`` profiles and forward-model them. Returns (res, app_res, phase)."""
        res = self.sample(n, seed)
        app, ph = forward.simulate(res, freqs, executor)
        assert np.all(np.isfinite(app)) and np.all(np.isfinite(ph)) and np.all(app > 0)
        return res, app, ph
