"""Shared machinery for the cluster experiments.

An experiment is a set of independent items (test samples or stations). Each
item is processed in its own worker process (forked, so the loaded data are
inherited rather than re-read), writes its own result file atomically, and is
skipped on resubmission, so a job killed by the time limit is simply resubmitted.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Hashable, List, Optional, Sequence

import numpy as np

_ACTIVE: Optional["Experiment"] = None      # set before the pool forks; read by workers


def _worker(item):
    return _ACTIVE._process(item)


class Experiment(ABC):
    def __init__(self, out_dir: str, resume: bool = True, workers: Optional[int] = None):
        self.out_dir, self.resume, self.workers = out_dir, resume, workers

    # ---- to implement --------------------------------------------------------
    @abstractmethod
    def item_path(self, item: Hashable) -> str: ...

    @abstractmethod
    def run_item(self, item: Hashable) -> dict:
        """Compute one item; return the dictionary to store."""

    @abstractmethod
    def summary_line(self, record: dict) -> str: ...

    @abstractmethod
    def collect(self, items: Sequence[Hashable]) -> None:
        """Aggregate finished items into tables and figures."""

    # ---- machinery -------------------------------------------------------------
    def _process(self, item):
        try:
            rec = self.run_item(item)
            final = self.item_path(item)
            tmp = final[:-4] + ".tmp.npz"
            np.savez(tmp, **rec)
            os.replace(tmp, final)          # atomic: a killed job never leaves a partial file
            return dict(ok=True, item=item, msg=self.summary_line(rec))
        except Exception:
            return dict(ok=False, item=item, msg=traceback.format_exc())

    def pending(self, items: Sequence[Hashable]) -> List[Hashable]:
        return [i for i in items if not (self.resume and os.path.exists(self.item_path(i)))]

    def n_workers(self, n_todo: int) -> int:
        n = self.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        return max(1, min(n, n_todo)) if n_todo else 1

    def run(self, items: Sequence[Hashable]) -> List[Hashable]:
        """Process all pending items; returns the failed ones."""
        global _ACTIVE
        todo = self.pending(items)
        nw = self.n_workers(len(todo))
        print(f"{len(items)} requested, {len(items) - len(todo)} already done, "
              f"{len(todo)} to run on {nw} worker process(es)", flush=True)
        failed, t0 = [], time.time()
        if todo:
            _ACTIVE = self
            can_fork = "fork" in mp.get_all_start_methods()
            if can_fork and nw > 1:
                with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("fork")) as pool:
                    results = (f.result() for f in as_completed([pool.submit(_worker, i) for i in todo]))
                    failed = self._report(results, len(todo))
            else:
                failed = self._report((self._process(i) for i in todo), len(todo))
        print(f"finished in {(time.time() - t0) / 60:.1f} min; failed: {failed or 'none'}", flush=True)
        return failed

    @staticmethod
    def _report(results, n_total) -> list:
        failed = []
        for n, res in enumerate(results, 1):
            if res["ok"]:
                print(f"[{n}/{n_total}] {res['msg']}", flush=True)
            else:
                failed.append(res["item"])
                print(f"[{n}/{n_total}] {res['item']} FAILED:\n{res['msg']}", flush=True)
        return failed
