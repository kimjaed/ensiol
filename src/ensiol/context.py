"""One object that wires the components together.

``ExperimentContext`` holds the grid, the prior catalog, the whitening, the
calibrated ``d_min`` and every component built from them. Experiments and
notebooks both start from it, so a notebook runs the same code path as the
cluster scripts.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .baselines import PCNConfig, PCNSampler
from .catalog import Catalog, MinDistanceCalibrator, Whitener
from .config import EnsIOLConfig, ScoringConfig
from .data import Grid, KilaueaDataset, Observation, PriorEnsemble
from .enscgp import EnsCGP
from .forward import SerialExecutor
from .growth import EnsIOL, GrowthResult
from .metrics import Scorer
from .snapshots import SnapshotEvaluator


def station_seed(base_seed: int, station_id: str) -> int:
    """Deterministic per-station seed. ``zlib.crc32`` rather than ``hash``,
    because Python randomises string hashes per process."""
    return base_seed + zlib.crc32(station_id.encode()) % (2 ** 31)


@dataclass
class AugmentationRun:
    """Result of running EnsIOL for one observation.

    ``base`` is the starting catalog (after the leakage guard, if a truth was
    given); ``catalog`` is the grown catalog.
    """

    context: "ExperimentContext"
    observation: Observation
    seed: int
    truth_linear: Optional[np.ndarray]
    base: Catalog
    catalog: Catalog
    growth: GrowthResult
    n_leak_removed: int
    runtime_s: float

    @property
    def n0(self) -> int:
        return self.base.n

    @property
    def n_added(self) -> int:
        return self.catalog.n - self.base.n

    @property
    def additions(self):
        return self.catalog.additions_since(self.n0)

    @property
    def truth_log10(self) -> Optional[np.ndarray]:
        return None if self.truth_linear is None else np.log10(self.truth_linear)

    def evolution_rows(self, iterations: Optional[List[int]] = None, n_rows: Optional[int] = None,
                       executor=None) -> List[dict]:
        """Single backward EnsCGP updates at chosen iterations (row 0 = original catalog)."""
        ctx = self.context
        snap = ctx.snapshots if n_rows is None else SnapshotEvaluator(ctx.enscgp, n_rows)
        if iterations is None:
            iterations = snap.row_iterations(self.growth)
        return snap.rows(self.base, self.additions, self.growth.cumulative_calls, self.observation,
                         self.seed, executor or SerialExecutor(), iterations)


class ExperimentContext:
    def __init__(self, grid: Grid, prior: PriorEnsemble, config: EnsIOLConfig = EnsIOLConfig(),
                 scoring: ScoringConfig = ScoringConfig(), pcn: PCNConfig = PCNConfig(), verbose: bool = True):
        self.grid, self.prior, self.config, self.scoring, self.pcn = grid, prior, config, scoring, pcn
        self.forward = grid.forward_operator()
        self.whitener = Whitener.from_labels(prior.labels)
        self.d_min = MinDistanceCalibrator(config.d_min_quantile, config.d_min_frac, config.n_probe,
                                           config.calibration_seed).calibrate(prior.labels, self.whitener)
        self.enscgp = EnsCGP(self.forward, config)
        self.ensiol = EnsIOL(self.forward, config, self.whitener, self.d_min)
        self.scorer = Scorer(self.forward, self.enscgp, scoring)
        self.snapshots = SnapshotEvaluator(self.enscgp)
        self.sampler = PCNSampler(self.forward, config, self.whitener, self.d_min, pcn)
        self.test = None
        if verbose:
            print(f"prior catalog: {len(prior)} members | {grid.n_layers} layers | "
                  f"{grid.freqs.size} freqs | d_min = {self.d_min:.4f}", flush=True)

    @classmethod
    def from_data_dir(cls, data_dir: str, with_test: bool = False, **kw) -> "ExperimentContext":
        ds = KilaueaDataset.load(data_dir, with_test=with_test)
        ctx = cls(ds.grid, ds.prior, **kw)
        ctx.test = ds.test
        return ctx

    def starting_catalog(self, truth_linear=None):
        """Fresh copy of the prior catalog; members identical to ``truth_linear``
        are dropped (leakage guard). Returns (catalog, n_removed)."""
        drop = None if truth_linear is None else self.prior.duplicate_mask(truth_linear)
        L, A, P = self.prior.arrays(drop)
        return Catalog(L, A, P, self.grid.freqs), (0 if drop is None else int(drop.sum()))

    def augment(self, obs: Observation, seed: int, truth_linear=None, executor=None) -> AugmentationRun:
        """Run EnsIOL on one observation, starting from a fresh copy of the prior catalog."""
        executor = executor or SerialExecutor()
        base, n_removed = self.starting_catalog(truth_linear)
        cat = base.copy()
        t0 = time.time()
        growth = self.ensiol.grow(obs, cat, seed, executor)
        return AugmentationRun(self, obs, seed, truth_linear, base, cat, growth, n_removed, time.time() - t0)
