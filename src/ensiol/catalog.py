"""The growing catalog of (model, response) pairs and its geometry."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .forward import match_or_interpolate


class Whitener:
    """Per-layer scaling by the starting catalog's standard deviation.

    Distances between models are computed in this whitened space.
    """

    def __init__(self, ref_std: np.ndarray):
        self.ref_std = np.asarray(ref_std, float)

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> "Whitener":
        """``labels``: natural-log resistivity, (n_layers, n)."""
        return cls(np.maximum(np.std(labels, axis=1), 1e-8))

    def apply(self, labels: np.ndarray) -> np.ndarray:
        """(n_layers, n) -> (n, n_layers)."""
        return (labels / self.ref_std[:, None]).T

    def apply_vector(self, x: np.ndarray) -> np.ndarray:
        return (x / self.ref_std).reshape(1, -1)


class Catalog:
    """Models and responses stored column-wise.

    Attributes
    ----------
    Labels : (n_layers, n) natural-log resistivity
    AppRes_log : (n_freqs_cat, n) natural-log apparent resistivity on the catalog grid
    Phase : (n_freqs_cat, n) degrees on the catalog grid

    The responses resampled to an observation's frequencies are cached and
    extended incrementally when members are appended.
    """

    def __init__(self, Labels, AppRes_log, Phase, ens_freqs):
        self.Labels = Labels
        self.AppRes_log = AppRes_log
        self.Phase = Phase
        self.ens_freqs = ens_freqs
        self.n_original = Labels.shape[1]
        self._real_freqs = None
        self._AppRes_real = None
        self._Phase_real = None

    @property
    def n(self) -> int:
        return self.Labels.shape[1]

    def copy(self) -> "Catalog":
        return Catalog(self.Labels.copy(), self.AppRes_log.copy(), self.Phase.copy(), self.ens_freqs)

    def head(self, m: int) -> "Catalog":
        """The catalog as it stood when it had ``m`` members (members are only ever appended)."""
        return Catalog(self.Labels[:, :m], self.AppRes_log[:, :m], self.Phase[:, :m], self.ens_freqs)

    def with_additions(self, add_labels, add_app, add_phase, k: int) -> "Catalog":
        """This catalog plus the first ``k`` of the given additions."""
        return Catalog(np.hstack([self.Labels, add_labels[:, :k]]),
                       np.hstack([self.AppRes_log, add_app[:, :k]]),
                       np.hstack([self.Phase, add_phase[:, :k]]), self.ens_freqs)

    def additions_since(self, n0: int):
        return self.Labels[:, n0:], self.AppRes_log[:, n0:], self.Phase[:, n0:]

    def interp_to_real(self, real_freqs):
        if self._real_freqs is None or not np.array_equal(self._real_freqs, real_freqs):
            A, P, _, _ = match_or_interpolate(self.ens_freqs, real_freqs, self.AppRes_log, self.Phase)
            self._real_freqs, self._AppRes_real, self._Phase_real = real_freqs, A, P
        elif self._AppRes_real.shape[1] < self.n:
            k = self._AppRes_real.shape[1]
            A_new, P_new, _, _ = match_or_interpolate(self.ens_freqs, real_freqs,
                                                      self.AppRes_log[:, k:], self.Phase[:, k:])
            self._AppRes_real = np.hstack([self._AppRes_real, A_new])
            self._Phase_real = np.hstack([self._Phase_real, P_new])
        return self._AppRes_real, self._Phase_real

    def append(self, Labels_new, AppRes_log_new, Phase_new) -> None:
        self.Labels = np.hstack([self.Labels, Labels_new])
        self.AppRes_log = np.hstack([self.AppRes_log, AppRes_log_new])
        self.Phase = np.hstack([self.Phase, Phase_new])


class MinDistanceCalibrator:
    """Sets the non-redundancy distance ``d_min``.

    ``d_min`` is ``frac`` times the ``quantile`` of nearest-neighbour distances
    (in whitened space) among a random probe subset of the starting catalog.
    """

    def __init__(self, quantile: float = 0.5, frac: float = 0.7, n_probe: int = 2000, seed: int = 12345):
        self.quantile, self.frac, self.n_probe, self.seed = quantile, frac, n_probe, seed

    def calibrate(self, labels: np.ndarray, whitener: Whitener) -> float:
        rng = np.random.default_rng(self.seed)
        W = whitener.apply(labels)
        n_ens = W.shape[0]
        probe_idx = rng.choice(n_ens, size=min(self.n_probe, n_ens), replace=False)
        D = cdist(W[probe_idx], W, metric="euclidean")
        D[np.arange(len(probe_idx)), probe_idx] = np.inf
        nn_dist = D.min(axis=1)
        return float(self.frac * np.quantile(nn_dist, self.quantile))
