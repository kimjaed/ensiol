"""EnsIOL: incremental growth of the catalog against one observation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.spatial.distance import cdist

from .catalog import Catalog, Whitener
from .config import EnsIOLConfig
from .data import Observation
from .enscgp import EnsCGP
from .forward import MTForward1D


class CandidateSelector:
    """Chooses which candidate models join the catalog.

    A candidate is eligible if every layer lies inside the physical bounds and
    it is farther than ``d_min`` (whitened Euclidean distance) from every
    catalog member. Eligible candidates are taken in order of distance from the
    candidates' mean (``"targeted"``) or in random order (``"random"``), up to
    ``n_add_max``.
    """

    def __init__(self, config: EnsIOLConfig, whitener: Whitener, d_min: float):
        self.config, self.whitener, self.d_min = config, whitener, d_min

    def select(self, Labels_corr: np.ndarray, catalog: Catalog, rng=None) -> np.ndarray:
        cfg = self.config
        lo, hi = cfg.log_bounds
        Wc = self.whitener.apply(Labels_corr)
        d_from_mean = np.linalg.norm(Wc - Wc.mean(axis=0, keepdims=True), axis=1)
        if cfg.growth_mode == "targeted":
            order = np.argsort(d_from_mean)[::-1]
        elif cfg.growth_mode == "random":
            order = rng.permutation(Labels_corr.shape[1])
        else:
            raise ValueError(cfg.growth_mode)
        is_valid = ((Labels_corr >= lo) & (Labels_corr <= hi)).all(axis=0)
        valid_order = np.array([i for i in order if is_valid[i]], dtype=int)
        if valid_order.size == 0:
            return np.array([], dtype=int)
        d_to_cat = cdist(Wc[valid_order], self.whitener.apply(catalog.Labels), metric="euclidean").min(axis=1)
        return valid_order[d_to_cat > self.d_min][:cfg.n_add_max]


class SurrogateCorrector:
    """Forward/backward correction used inside the growth loop.

    The forward step approximates the response of the current mean model from
    the catalog itself (a local linear regression of responses on models over
    the nearest members), so it costs no forward calls. The backward step then
    shifts the ensemble by the misfit between the observation and that
    approximate response.
    """

    def __init__(self, config: EnsIOLConfig, whitener: Whitener):
        self.config, self.whitener, self.noise = config, whitener, config.noise

    def surrogate_response(self, x_q_hat, catalog: Catalog, obs: Observation, reg_eps: float = 1e-8):
        W_cat = self.whitener.apply(catalog.Labels)
        d = cdist(self.whitener.apply_vector(x_q_hat), W_cat, metric="euclidean")[0]
        k = min(self.config.n_neighbors_surrogate, catalog.n)
        nn_idx = np.argsort(d)[:k]
        X_e = catalog.Labels[:, nn_idx]
        cat_A, cat_P = catalog.interp_to_real(obs.freqs)
        Y_e = np.vstack([cat_A[:, nn_idx], cat_P[:, nn_idx]])
        dX = X_e - X_e.mean(axis=1, keepdims=True)
        dY = Y_e - Y_e.mean(axis=1, keepdims=True)
        C_XX = dX @ dX.T / (k - 1)
        C_YX = dY @ dX.T / (k - 1)
        C_XX_reg = C_XX.copy()
        C_XX_reg[np.diag_indices_from(C_XX_reg)] += reg_eps
        K_q = C_YX @ np.linalg.solve(C_XX_reg, np.eye(C_XX_reg.shape[0]))
        Y_e_plus = Y_e + K_q @ (x_q_hat[:, None] - X_e)
        return Y_e_plus.mean(axis=1), X_e, Y_e

    def correct(self, Labels_pred, X_e, Y_e, mean_response_approx, obs: Observation,
                damping: float, reg_eps: float = 1e-8):
        k = X_e.shape[1]
        n_freqs = obs.n_freqs
        noise_diag = np.concatenate([(self.noise.app_res_sigma ** 2) * np.ones(n_freqs),
                                     (self.noise.phase_sigma ** 2) * np.ones(n_freqs)])
        dX = X_e - X_e.mean(axis=1, keepdims=True)
        dY = Y_e - Y_e.mean(axis=1, keepdims=True)
        C_md = dX @ dY.T / (k - 1)
        C_dd = dY @ dY.T / (k - 1)
        A = C_dd.copy()
        A[np.diag_indices_from(A)] += noise_diag + reg_eps
        shift = C_md @ np.linalg.solve(A, obs.stacked - mean_response_approx)
        return Labels_pred + damping * shift[:, None], shift


@dataclass
class IterationRecord:
    iteration: int
    nRMS_approx: float
    mean_change: float
    n_accepted: int
    catalog_size: int


@dataclass
class GrowthResult:
    history: List[IterationRecord] = field(default_factory=list)
    exit_reason: str = "max_iter"

    @property
    def n_accepted(self) -> np.ndarray:
        """Members added at each iteration."""
        return np.array([h.n_accepted for h in self.history], int)

    @property
    def cumulative_calls(self) -> np.ndarray:
        """Forward calls spent after each iteration, with 0 prepended. Each
        accepted member costs exactly one forward call (the surrogate makes none)."""
        return np.concatenate([[0], np.cumsum(self.n_accepted)]).astype(int)

    @property
    def last_growing_iteration(self) -> int:
        """Last iteration that added members (trailing no-growth iterations dropped)."""
        cum = self.cumulative_calls
        last = len(cum) - 1
        while last > 0 and cum[last] == cum[last - 1]:
            last -= 1
        return last


class EnsIOL:
    """Ensemble incremental online learning.

    Each iteration (1) updates the members nearest to the observation
    (backward EnsCGP), (2) corrects that ensemble through the catalog's own
    surrogate response (forward/backward), (3) selects eligible candidates, and
    (4) forward-models them with the true physics and appends them to the
    catalog. The loop stops when no candidate is accepted or the ensemble mean
    stops changing.
    """

    def __init__(self, forward: MTForward1D, config: EnsIOLConfig, whitener: Whitener, d_min: float):
        self.forward, self.config, self.d_min = forward, config, d_min
        self.enscgp = EnsCGP(forward, config)
        self.selector = CandidateSelector(config, whitener, d_min)
        self.surrogate = SurrogateCorrector(config, whitener)
        self.whitener = whitener

    def grow(self, obs: Observation, catalog: Catalog, seed: int, executor) -> GrowthResult:
        """Grows ``catalog`` in place."""
        cfg = self.config
        rng = np.random.default_rng(seed)
        perturbation = self.enscgp.perturbation(obs.n_freqs, cfg.n_neighbors, rng)
        result, prev_mean = GrowthResult(), None

        for it in range(1, cfg.max_iter + 1):
            nn_idx, cat_A, cat_P = self.enscgp.neighbors(obs, catalog, cfg.n_neighbors)
            k = nn_idx.size
            Labels_pred = self.enscgp.update(obs, cat_A[:, nn_idx], cat_P[:, nn_idx], catalog.Labels[:, nn_idx],
                                             perturbation=perturbation[:, :k], damping=cfg.damping_loop)
            x_q_hat = Labels_pred.mean(axis=1)
            mean_resp, X_e, Y_e = self.surrogate.surrogate_response(x_q_hat, catalog, obs)
            nrms_approx = self.enscgp.nrms(obs, mean_resp[:obs.n_freqs], mean_resp[obs.n_freqs:])
            Labels_corr, _ = self.surrogate.correct(Labels_pred, X_e, Y_e, mean_resp, obs, cfg.damping_loop)

            mean_corr = np.mean(Labels_corr, axis=1)
            mean_change = (np.inf if prev_mean is None else
                           float(np.sqrt(np.mean(((mean_corr - prev_mean) / self.whitener.ref_std) ** 2))))
            prev_mean = mean_corr

            sel = self.selector.select(Labels_corr, catalog, rng=rng)
            n_accepted = len(sel)
            if n_accepted > 0:
                L_new = Labels_corr[:, sel]
                A_new, P_new, _, _ = self.forward.simulate_two_grids(L_new, catalog.ens_freqs, obs.freqs, executor)
                catalog.append(L_new, A_new, P_new)

            result.history.append(IterationRecord(it, nrms_approx, mean_change, n_accepted, catalog.n))
            if n_accepted < cfg.min_accepted:
                result.exit_reason = "growth_stall"
                break
            if mean_change < cfg.mean_tol:
                result.exit_reason = "mean_stable"
                break
        return result
