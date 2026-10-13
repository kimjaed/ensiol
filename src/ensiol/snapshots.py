"""Evaluating the catalog at successive stages of EnsIOL growth."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .catalog import Catalog
from .data import Observation
from .enscgp import EnsCGP
from .growth import GrowthResult


class SnapshotEvaluator:
    """Single backward EnsCGP updates against the catalog as it stood after
    chosen EnsIOL iterations.

    The catalog only ever grows by appending, so the catalog after iteration
    ``i`` is the original catalog plus the first ``cumulative_calls[i]``
    additions; every snapshot is rebuilt exactly from those.
    """

    def __init__(self, enscgp: EnsCGP, n_rows: int = 5):
        self.enscgp, self.n_rows = enscgp, n_rows

    def row_iterations(self, growth: GrowthResult) -> List[int]:
        """``n_rows`` evenly spaced iterations from 0 (original catalog) to the
        last iteration that added members."""
        last = growth.last_growing_iteration
        return sorted(set(np.round(np.linspace(0, last, self.n_rows)).astype(int).tolist()))

    def rows(self, base: Catalog, additions, cumulative_calls: np.ndarray, obs: Observation,
             seed: int, executor, iterations: Optional[Sequence[int]] = None) -> List[dict]:
        """One dict per iteration: iteration, catalog_size, n_added and the estimate's fields.

        ``additions`` is the (Labels, AppRes_log, Phase) tuple of the members
        added by EnsIOL, in order.
        """
        iterations = list(iterations)
        rows = []
        for it in iterations:
            k = int(cumulative_calls[it])
            snap = base.with_additions(*additions, k=k)
            est = self.enscgp.estimate(obs, snap, executor, seed=seed)
            rows.append(dict(iteration=it, catalog_size=snap.n, n_added=k, **est.as_dict()))
        return rows
