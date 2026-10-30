"""Baselines for catalog growth: an MCMC sampler and a naive principal-component grid.

A preconditioned Crank-Nicolson (pCN) sampler targets the posterior whose prior
is a Gaussian in natural-log resistivity with the starting catalog's own mean
and covariance, truncated to the same physical bounds EnsIOL uses. Accepted
states that are non-redundant (farther than ``d_min`` from every catalog
member) join the catalog, exactly as EnsIOL's candidates do, so both methods
start from the same information and are scored by the same evaluator.

The second baseline, :class:`PCGridSweep`, is deliberately naive: a regular grid over the leading
principal components of the original catalog, with no information from the observation.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .catalog import Catalog, Whitener
from .config import EnsIOLConfig
from .data import Observation
from .enscgp import EnsCGP
from .forward import MTForward1D, SerialExecutor
from .metrics import Scorer


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


# =============================================================================================
# Naive baseline: a regular grid over the leading principal components of the catalog
# =============================================================================================
@dataclass(frozen=True)
class PCGridConfig:
    """Settings of the naive principal-component grid sweep.

    ``grid_sizes`` are the numbers of values per axis; each size is its own grid and its own point
    on the cost curve (a grid with k points per axis is not a prefix of a larger grid).
    """

    n_modes: int = 4
    grid_sizes: Tuple[int, ...] = (3, 4, 5, 6, 7, 8, 10, 12, 13)
    chunk: int = 500                   # rows per distance block, to bound memory

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PCBasis:
    """Mean and leading principal components of the original catalog, in natural-log resistivity.

    A model of the sweep is ``mu + U a``: the catalog mean plus the ``n_modes`` fixed depth patterns
    in the columns of ``U``, each scaled by a coefficient. The coefficient of each pattern ranges
    between the smallest and the largest value the catalog itself has on that pattern.
    """

    mu: np.ndarray            # (n_layers, 1)
    U: np.ndarray             # (n_layers, n_modes)
    cmin: np.ndarray          # (n_modes,) smallest catalog coefficient on each pattern
    cmax: np.ndarray          # (n_modes,) largest catalog coefficient on each pattern
    var_each: np.ndarray      # (n_modes,) share of the catalog variance of each pattern

    @property
    def n_modes(self) -> int:
        return self.U.shape[1]

    @property
    def var_explained(self) -> float:
        return float(np.sum(self.var_each))

    @classmethod
    def fit(cls, labels: np.ndarray, n_modes: int) -> "PCBasis":
        """``labels``: natural-log resistivity of the whole original catalog, (n_layers, n_members)."""
        mu = labels.mean(axis=1, keepdims=True)
        U, S, _ = np.linalg.svd(labels - mu, full_matrices=False)
        U = U[:, :n_modes]
        coef = U.T @ (labels - mu)                                  # (n_modes, n_members)
        var = S ** 2 / np.sum(S ** 2)
        return cls(mu, U, coef.min(axis=1), coef.max(axis=1), var[:n_modes])


@dataclass
class PCGrid:
    """One simulated grid: the models inside the physical bounds and their responses."""

    k: int                      # values per axis
    models: np.ndarray          # (n_layers, n_valid) natural-log resistivity
    app_res_log: np.ndarray     # (n_freqs, n_valid) natural-log apparent resistivity, catalog grid
    phase: np.ndarray           # (n_freqs, n_valid) degrees, catalog grid
    n_nominal: int              # k ** n_modes before the bounds are applied

    @property
    def n_valid(self) -> int:
        return self.models.shape[1]

    @property
    def calls(self) -> int:
        """Forward calls: every model inside the bounds costs exactly one."""
        return self.n_valid


class PCGridSweep:
    """Naive catalog growth: sweep a regular grid over the leading principal components.

    The grid uses no information from the observation, so every grid is simulated once and shared by
    all test samples; each sample is still charged the full forward calls of the grid, exactly as if it
    had run the sweep itself. For each sample and grid size, a grid model joins the sample's starting
    catalog if it is farther than ``d_min`` (whitened distance) from every member of that starting
    catalog, the same non-redundancy rule as for EnsIOL's candidates and MCMC's states. The grown
    catalog is scored by the same single backward EnsCGP update as every other method.
    """

    def __init__(self, forward: MTForward1D, config: EnsIOLConfig, whitener: Whitener, d_min: float,
                 scorer: Scorer, enscgp: EnsCGP, freqs: np.ndarray, prior_labels: np.ndarray,
                 sweep: PCGridConfig = PCGridConfig()):
        self.forward, self.config, self.whitener, self.d_min = forward, config, whitener, d_min
        self.scorer, self.enscgp, self.freqs, self.sweep = scorer, enscgp, freqs, sweep
        self._prior_labels = prior_labels
        self._basis: Optional[PCBasis] = None
        self.grids: Dict[int, PCGrid] = {}

    @property
    def basis(self) -> PCBasis:
        """PCA of the original (full) prior catalog; independent of any observation."""
        if self._basis is None:
            self._basis = PCBasis.fit(self._prior_labels, self.sweep.n_modes)
        return self._basis

    def describe_basis(self) -> str:
        b = self.basis
        return (f"PCA: {b.n_modes} modes explain {100 * b.var_explained:.1f}% of the catalog variance; "
                "coefficient ranges " + ", ".join(f"[{lo:.1f}, {hi:.1f}]" for lo, hi in zip(b.cmin, b.cmax)))

    def build_grids(self, grid_sizes: Optional[Sequence[int]] = None, executor=None, verbose: bool = True) -> None:
        """Simulate every requested grid that has not been simulated yet.

        Models outside the physical bounds are dropped before simulation and cost nothing.
        """
        ex = executor or SerialExecutor()
        sizes = list(grid_sizes) if grid_sizes is not None else list(self.sweep.grid_sizes)
        b = self.basis
        lo, hi = self.config.log_bounds
        for k in sizes:
            if k in self.grids:
                continue
            axes = [np.linspace(b.cmin[j], b.cmax[j], k) for j in range(b.n_modes)]
            coef = np.array(list(itertools.product(*axes))).T          # (n_modes, k ** n_modes)
            models = b.mu + b.U @ coef                                  # (n_layers, k ** n_modes)
            ok = np.all((models >= lo) & (models <= hi), axis=0)
            models = models[:, ok]
            t0 = time.time()
            if models.shape[1]:
                app, ph = self.forward.simulate(np.exp(models.T), self.freqs, ex)
                app_log, phase = np.log(app.T), ph.T
            else:
                app_log, phase = np.empty((self.freqs.size, 0)), np.empty((self.freqs.size, 0))
            self.grids[k] = PCGrid(k, models, app_log, phase, int(coef.shape[1]))
            if verbose:
                print(f"  grid k={k:2d}: {coef.shape[1]:6d} nominal, {int(ok.sum()):6d} inside the bounds "
                      f"(= forward calls)  [{time.time() - t0:.0f} s]", flush=True)

    def _admitted(self, W_new: np.ndarray, W_cat: np.ndarray) -> np.ndarray:
        """Grid models farther than d_min from every member of the starting catalog."""
        keep = np.ones(W_new.shape[0], bool)
        c = self.sweep.chunk
        for i in range(0, W_new.shape[0], c):
            keep[i:i + c] = cdist(W_new[i:i + c], W_cat).min(axis=1) > self.d_min
        return keep

    def _neighbour_share(self, catalog: Catalog, obs: Observation, n_original: int) -> float:
        """Share of the members used by the evaluation update that come from the sweep."""
        nn_idx, _, _ = self.enscgp.neighbors(obs, catalog, self.config.n_neighbors)
        return float(np.mean(nn_idx >= n_original))

    def evaluate(self, base: Catalog, obs: Observation, truth_log10: np.ndarray, seed: int, executor=None,
                 grid_sizes: Optional[Sequence[int]] = None) -> dict:
        """Score the sweep for one test sample, one point per grid size.

        ``base`` is the sample's starting catalog (leakage guard applied). Returns arrays ``grid_*``
        with one entry per grid size and the scores of the original catalog, ``orig_*``.
        """
        ex = executor or SerialExecutor()
        ks = list(grid_sizes) if grid_sizes is not None else list(self.sweep.grid_sizes)
        self.build_grids([k for k in ks if k not in self.grids], ex)
        n0 = base.n
        W0 = self.whitener.apply(base.Labels)
        s0 = self.scorer.score_catalog(base, obs, truth_log10, seed, ex)

        keys = ("rms", "nrms_rom", "nrms_mor", "U", "cov95")
        rows = {key: [] for key in keys}
        calls, added, share, mean_log10 = [], [], [], []
        for k in ks:
            g = self.grids[k]
            keep = self._admitted(self.whitener.apply(g.models), W0)
            n_add = int(keep.sum())
            cat = base.with_additions(g.models[:, keep], g.app_res_log[:, keep], g.phase[:, keep], k=n_add)
            s = self.scorer.score_catalog(cat, obs, truth_log10, seed, ex)
            for key in keys:
                rows[key].append(s[key])
            mean_log10.append(s["mean_log10"])
            calls.append(g.calls)
            added.append(n_add)
            share.append(self._neighbour_share(cat, obs, n0))

        out = dict(grid_sizes=np.array(ks), grid_calls=np.array(calls), grid_added=np.array(added),
                   grid_nbr_share=np.array(share), grid_mean_log10=np.array(mean_log10))
        for key in keys:
            out[f"grid_{key}"] = np.array(rows[key])
            out[f"orig_{key}"] = s0[key]
        return out
