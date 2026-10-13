"""Ensemble conditional Gaussian process (EnsCGP) estimate from a catalog."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist

from .catalog import Catalog
from .config import EnsIOLConfig
from .data import Observation
from .forward import MTForward1D


@dataclass
class Estimate:
    """Ensemble produced by one EnsCGP update, with its simulated responses."""

    Labels_est: np.ndarray       # (n_layers, k) natural-log resistivity
    AppRes_real: np.ndarray      # (n_obs_freqs, k) natural-log apparent resistivity, true forward
    Phase_real: np.ndarray       # (n_obs_freqs, k) degrees
    nRMS: float                  # misfit of the mean of the members' responses
    n_neighbors_used: int

    @classmethod
    def from_dict(cls, d: dict) -> "Estimate":
        return cls(d["Labels_est"], d["AppRes_real"], d["Phase_real"], d["nRMS"], d["n_neighbors_used"])

    def as_dict(self) -> dict:
        return dict(Labels_est=self.Labels_est, AppRes_real=self.AppRes_real, Phase_real=self.Phase_real,
                    nRMS=self.nRMS, n_neighbors_used=self.n_neighbors_used)


class EnsCGP:
    """Backward update of the members of a catalog nearest to an observation.

    The update moves the ``n_neighbors`` catalog members closest to the
    observation in data space (cosine distance) toward the observation,
    using the covariances of the members themselves.
    """

    def __init__(self, forward: MTForward1D, config: EnsIOLConfig):
        self.forward = forward
        self.config = config
        self.noise = config.noise

    # ---- small helpers ---------------------------------------------------
    def perturbation(self, n_freqs: int, n_members: int, rng: np.random.Generator) -> np.ndarray:
        """Observation perturbations, shape (2 * n_freqs, n_members)."""
        std = np.concatenate([self.noise.app_res_sigma * np.ones(n_freqs),
                              self.noise.phase_sigma * np.ones(n_freqs)])[:, None]
        return std * rng.standard_normal((2 * n_freqs, n_members))

    def nrms(self, obs: Observation, mean_app_log, mean_phase) -> float:
        r_a = (obs.app_res_log - mean_app_log) / self.noise.app_res_sigma
        r_p = (obs.phase - mean_phase) / self.noise.phase_sigma
        return float(np.sqrt(np.mean(np.concatenate([r_a, r_p]) ** 2)))

    def neighbors(self, obs: Observation, catalog: Catalog, k: int):
        """Indices of the k catalog members closest to the observation, and the
        catalog responses resampled to the observation's frequencies."""
        cat_A, cat_P = catalog.interp_to_real(obs.freqs)
        ens_data = np.vstack([cat_A, cat_P])
        dist = cdist(obs.stacked.reshape(1, -1), ens_data.T, metric="cosine")[0]
        return np.argsort(dist)[:min(k, catalog.n)], cat_A, cat_P

    # ---- the update itself -------------------------------------------------
    def update(self, obs: Observation, AppRes_nn, Phase_nn, Labels_nn,
               perturbation=None, damping: float = 1.0, reg_eps: float = 1e-12) -> np.ndarray:
        """One backward update of ``Labels_nn`` toward the observation."""
        Labels = Labels_nn.copy()      # contiguous copy: keeps BLAS kernels (and rounding) identical run to run
        n_freqs, N = AppRes_nn.shape[0], Labels.shape[1]
        noise_diag = np.concatenate([(self.noise.app_res_sigma ** 2) * np.ones(n_freqs),
                                     (self.noise.phase_sigma ** 2) * np.ones(n_freqs)])
        noise_diag = np.maximum(noise_diag, 1e-30)
        Data_current = np.vstack([AppRes_nn, Phase_nn])
        y = obs.stacked[:, None]
        Y = y + perturbation if perturbation is not None else np.repeat(y, N, axis=1)
        dM = Labels - np.mean(Labels, axis=1, keepdims=True)
        dD = Data_current - np.mean(Data_current, axis=1, keepdims=True)
        C_md = (dM @ dD.T) / (N - 1)
        C_dd = (dD @ dD.T) / (N - 1)
        A = C_dd.copy()
        A[np.diag_indices_from(A)] += noise_diag + reg_eps
        return Labels + damping * (C_md @ solve(A, Y - Data_current))

    def estimate(self, obs: Observation, catalog: Catalog, executor, seed: int = 0,
                 damping: float = None) -> Estimate:
        """Evaluation estimate: one backward update against ``catalog``.

        No growth and no correction step. The updated members are simulated
        with the true forward operator only to report their responses; those
        responses are not fed back into the estimate.
        """
        damping = self.config.damping_eval if damping is None else damping
        rng = np.random.default_rng(seed)
        nn_idx, cat_A, cat_P = self.neighbors(obs, catalog, self.config.n_neighbors)
        k = nn_idx.size
        pert = self.perturbation(obs.n_freqs, k, rng)
        Labels_est = self.update(obs, cat_A[:, nn_idx], cat_P[:, nn_idx], catalog.Labels[:, nn_idx],
                                 perturbation=pert, damping=damping)
        _, _, A_real, P_real = self.forward.simulate_two_grids(Labels_est, catalog.ens_freqs, obs.freqs, executor)
        nrms = self.nrms(obs, np.mean(A_real, axis=1), np.mean(P_real, axis=1))
        return Estimate(Labels_est, A_real, P_real, nrms, k)
