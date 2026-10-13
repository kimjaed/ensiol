"""MCMC baseline for catalog growth.

A preconditioned Crank-Nicolson (pCN) sampler targets the posterior whose prior
is a Gaussian in natural-log resistivity with the starting catalog's own mean
and covariance, truncated to the same physical bounds EnsIOL uses. Accepted
states that are non-redundant (farther than ``d_min`` from every catalog
member) join the catalog, exactly as EnsIOL's candidates do, so both methods
start from the same information and are scored by the same evaluator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .catalog import Catalog, Whitener
from .config import EnsIOLConfig
from .data import Observation
from .enscgp import EnsCGP
from .forward import MTForward1D


@dataclass(frozen=True)
class PCNConfig:
    n_chains: int = 50
    max_calls: int = 20000
    beta_init: float = 0.1
    beta_bounds: Tuple[float, float] = (1e-3, 1.0)
    target_accept: float = 0.25
    adapt_every: int = 10              # rounds
    adapt_until_calls: int = 5000      # beta frozen after this many forward calls
    cov_jitter: float = 1e-6           # relative jitter on the prior covariance diagonal
    seed_offset: int = 10_000_000      # keeps the MCMC stream independent of EnsIOL's

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class McmcResult:
    add_L: np.ndarray
    add_A: np.ndarray
    add_P: np.ndarray
    calls_per_round: np.ndarray
    n_added_cum: np.ndarray
    acc: np.ndarray
    beta: np.ndarray

    @property
    def total_calls(self) -> int:
        return int(self.calls_per_round[-1])

    @property
    def total_added(self) -> int:
        return int(self.n_added_cum[-1])

    def at_budget(self, budget: int) -> Tuple[int, int]:
        """(forward calls actually used, members added) after the rounds
        completed within ``budget`` calls."""
        r_done = int(np.searchsorted(self.calls_per_round, budget, side="right"))
        if r_done == 0:
            return 0, 0
        return int(self.calls_per_round[r_done - 1]), int(self.n_added_cum[r_done - 1])


class PCNSampler:
    def __init__(self, forward: MTForward1D, config: EnsIOLConfig, whitener: Whitener, d_min: float,
                 pcn: PCNConfig = PCNConfig()):
        self.forward, self.config, self.whitener, self.d_min, self.pcn = forward, config, whitener, d_min, pcn
        self.enscgp = EnsCGP(forward, config)

    def log_likelihood(self, A_log, P, obs: Observation) -> np.ndarray:
        noise = self.config.noise
        ra = (obs.app_res_log[:, None] - A_log) / noise.app_res_sigma
        rp = (obs.phase[:, None] - P) / noise.phase_sigma
        return -0.5 * (np.sum(ra ** 2, axis=0) + np.sum(rp ** 2, axis=0))

    def run(self, catalog: Catalog, obs: Observation, seed: int, executor) -> McmcResult:
        """Sample from the posterior with ``catalog`` (the original, ungrown one) as the prior."""
        pcn, cfg = self.pcn, self.config
        lo, hi = cfg.log_bounds
        rng = np.random.default_rng(seed + pcn.seed_offset)
        Labels0 = catalog.Labels
        n_layers = Labels0.shape[0]

        mu = Labels0.mean(axis=1)
        C = np.cov(Labels0)
        C[np.diag_indices_from(C)] += pcn.cov_jitter * np.mean(np.diag(C))
        L = np.linalg.cholesky(C)

        # starting states: members of the data-space neighbour set EnsIOL's first step uses
        nn_idx, cA, cP = self.enscgp.neighbors(obs, catalog, cfg.n_neighbors)
        start = rng.choice(nn_idx, size=pcn.n_chains, replace=False)
        cur = Labels0[:, start].copy()
        cur_A, cur_P = cA[:, start].copy(), cP[:, start].copy()      # catalog responses: no new calls
        cur_ll = self.log_likelihood(cur_A, cur_P, obs)

        catalog_W = self.whitener.apply(Labels0)                      # grows with additions
        add_L, add_A, add_P = [], [], []
        calls, beta = 0, pcn.beta_init
        calls_per_round, n_added_cum, acc_hist, beta_hist, recent_acc = [], [], [], [], []

        while calls < pcn.max_calls:
            xi = rng.standard_normal((n_layers, pcn.n_chains))
            prop = mu[:, None] + np.sqrt(1 - beta ** 2) * (cur - mu[:, None]) + beta * (L @ xi)
            valid = np.all((prop >= lo) & (prop <= hi), axis=0)
            vidx = np.where(valid)[0]
            accepted = np.zeros(pcn.n_chains, bool)
            if vidx.size:
                A_ens, P_ens, A_obs, P_obs = self.forward.simulate_two_grids(
                    prop[:, vidx], catalog.ens_freqs, obs.freqs, executor)
                calls += vidx.size
                ll = self.log_likelihood(A_obs, P_obs, obs)
                acc = np.log(rng.random(vidx.size)) < (ll - cur_ll[vidx])
                aidx = vidx[acc]
                if aidx.size:
                    cur[:, aidx] = prop[:, aidx]
                    cur_ll[aidx] = ll[acc]
                    accepted[aidx] = True
                    # catalog growth: same d_min rule as EnsIOL (checked against the catalog
                    # only, not among candidates of the same round)
                    Wn = self.whitener.apply(prop[:, aidx])
                    kr = np.where(cdist(Wn, catalog_W).min(axis=1) > self.d_min)[0]
                    if kr.size:
                        add_L.append(prop[:, aidx[kr]])
                        add_A.append(A_ens[:, acc][:, kr])
                        add_P.append(P_ens[:, acc][:, kr])
                        catalog_W = np.vstack([catalog_W, Wn[kr]])
            rate = accepted.mean()
            recent_acc.append(rate)
            if calls <= pcn.adapt_until_calls and len(recent_acc) % pcn.adapt_every == 0:
                r = np.mean(recent_acc[-pcn.adapt_every:])
                if r > pcn.target_accept + 0.05:
                    beta = min(beta * 1.3, pcn.beta_bounds[1])
                elif r < pcn.target_accept - 0.05:
                    beta = max(beta * 0.7, pcn.beta_bounds[0])
            calls_per_round.append(calls)
            n_added_cum.append(sum(a.shape[1] for a in add_L))
            acc_hist.append(rate)
            beta_hist.append(beta)

        stack = lambda lst, nrow: np.hstack(lst) if lst else np.empty((nrow, 0))
        return McmcResult(stack(add_L, n_layers), stack(add_A, len(catalog.ens_freqs)),
                          stack(add_P, len(catalog.ens_freqs)), np.array(calls_per_round),
                          np.array(n_added_cum), np.array(acc_hist), np.array(beta_hist))
