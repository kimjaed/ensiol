"""Synthetic test set for a new geologic setting.

Models are drawn from a Gaussian with a structured mean (a layered resistivity
structure loosely motivated by a volcanic setting) and a depth-dependent
standard deviation, deliberately different from the Gabbs Valley prior.
Two independent splits come from child streams of one ``SeedSequence``: an
adaptation split (observations assimilated by EnsIOL) and an optional held-out
split. Both are noiseless here; observation noise is applied downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .gmrf import smoothing_operator


@dataclass
class NewSettingSpec:
    mean_anchors: List[Tuple[float, float]] = field(default_factory=lambda: [
        (10.0, 800.0),      # unsaturated basalt near the surface
        (250.0, 800.0),
        (700.0, 30.0),      # transition through the water table
        (1100.0, 8.0),      # conductive saturated / altered zone
        (1800.0, 8.0),
        (3200.0, 150.0),
        (4500.0, 400.0),    # resistive dense intrusive rock
        (7000.0, 400.0),
        (11000.0, 150.0),   # moderate decrease at depth
    ])
    mean_smooth_cells: float = 1.5
    sigma_shallow_log10: float = 0.50
    sigma_deep_log10: float = 0.90
    sigma_transition_m: float = 2500.0
    sigma_transition_width_dec: float = 0.25
    corr_len_cells: float = 4.0
    res_bounds: Tuple[float, float] = (5e-3, 5e5)
    max_rejection_rounds: int = 50

    def mean_log_rho(self, depth_m) -> np.ndarray:
        """Mean in natural-log resistivity from the anchors (linear in log10 depth,
        clamped beyond the first and last anchor, lightly smoothed)."""
        z_a = np.log10([a[0] for a in self.mean_anchors])
        r_a = np.log10([a[1] for a in self.mean_anchors])
        log10_rho = np.interp(np.log10(depth_m), z_a, r_a)
        if self.mean_smooth_cells > 0:
            log10_rho = smoothing_operator(len(depth_m), self.mean_smooth_cells) @ log10_rho
        return log10_rho * np.log(10)

    def sigma_profile(self, depth_m) -> np.ndarray:
        """Depth-dependent standard deviation in natural-log resistivity."""
        t = 0.5 * (1 + np.tanh((np.log10(depth_m) - np.log10(self.sigma_transition_m)) / self.sigma_transition_width_dec))
        return (self.sigma_shallow_log10 + (self.sigma_deep_log10 - self.sigma_shallow_log10) * t) * np.log(10)

    def covariance(self, depth_m) -> np.ndarray:
        n = len(depth_m)
        idx = np.arange(n)
        R = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / self.corr_len_cells) ** 2)
        s = self.sigma_profile(depth_m)
        return s[:, None] * R * s[None, :]

    def sample(self, depth_interfaces, n_adapt: int = 100, n_holdout: int = 50, seed: int = 2026):
        """Returns (rho_adapt, rho_holdout) in ohm-m."""
        depth = 0.5 * (np.asarray(depth_interfaces)[:-1] + np.asarray(depth_interfaces)[1:])
        mean = self.mean_log_rho(depth)
        L = np.linalg.cholesky(self.covariance(depth) + 1e-8 * np.eye(depth.size))
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        lo, hi = self.res_bounds

        def draw(n, rng):
            return np.exp(mean[None, :] + rng.standard_normal((n, depth.size)) @ L.T)

        def with_rejection(n, rng):
            if n == 0:
                return np.empty((0, depth.size))
            acc, needed, rounds = [], n, 0
            while needed > 0 and rounds < self.max_rejection_rounds:
                cand = draw(needed, rng)
                ok = np.all((cand >= lo) & (cand <= hi), axis=1)
                acc.append(cand[ok]); needed -= int(ok.sum()); rounds += 1
            if needed > 0:
                raise RuntimeError(f"{needed} profile(s) still out of bounds after {rounds} rounds")
            return np.vstack(acc)[:n]

        return with_rejection(n_adapt, rngs[0]), with_rejection(n_holdout, rngs[1])
