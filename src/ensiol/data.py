"""Data containers and loaders.

Directory layout expected by :meth:`KilaueaDataset.load` (the file names are
those used throughout the project)::

    <data_dir>/kilauea_1d_ResModels_prior.txt   (n_prior, n_layers)  ohm-m
    <data_dir>/kilauea_1d_AppRes_prior.txt      (n_prior, n_freqs)   ohm-m
    <data_dir>/kilauea_1d_Phase_prior.txt       (n_prior, n_freqs)   degrees
    <data_dir>/kilauea_1d_freqs_prior.txt       (n_freqs,)           Hz
    <data_dir>/kilauea_depth_interfaces_m.txt   (n_layers + 1,)      m
    and, for the synthetic experiment, the same three *_test.txt files plus
    kilauea_1d_freqs_test.txt.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .forward import MTForward1D


@dataclass(frozen=True)
class Grid:
    """Depth interfaces (m) and catalog frequencies (Hz)."""

    depths: np.ndarray
    freqs: np.ndarray

    @property
    def n_layers(self) -> int:
        return self.depths.size - 1

    @property
    def depth_centers(self) -> np.ndarray:
        return 0.5 * (self.depths[:-1] + self.depths[1:])

    def forward_operator(self) -> MTForward1D:
        return MTForward1D(self.depths)


@dataclass
class Observation:
    """One MT sounding on its own frequency grid.

    ``app_res_log`` is natural-log apparent resistivity, ``phase`` is in degrees.
    """

    freqs: np.ndarray
    app_res_log: np.ndarray
    phase: np.ndarray

    @classmethod
    def from_linear(cls, freqs, app_res, phase) -> "Observation":
        return cls(np.asarray(freqs, float), np.log(np.asarray(app_res, float)),
                   np.asarray(phase, float))

    @property
    def n_freqs(self) -> int:
        return self.freqs.size

    @property
    def stacked(self) -> np.ndarray:
        """Data vector [log apparent resistivity, phase]."""
        return np.concatenate([self.app_res_log, self.phase])


@dataclass
class PriorEnsemble:
    """The starting (prior) catalog as stored on disk: one row per member."""

    res: np.ndarray          # (n, n_layers) ohm-m
    app_res: np.ndarray      # (n, n_freqs)  ohm-m
    phase: np.ndarray        # (n, n_freqs)  degrees

    @classmethod
    def load(cls, data_dir: str, suffix: str = "prior") -> "PriorEnsemble":
        return cls(np.loadtxt(f"{data_dir}/kilauea_1d_ResModels_{suffix}.txt"),
                   np.loadtxt(f"{data_dir}/kilauea_1d_AppRes_{suffix}.txt"),
                   np.loadtxt(f"{data_dir}/kilauea_1d_Phase_{suffix}.txt"))

    def __len__(self) -> int:
        return self.res.shape[0]

    @property
    def labels(self) -> np.ndarray:
        """Natural-log resistivity, shape (n_layers, n)."""
        return np.log(self.res).T

    def duplicate_mask(self, truth_linear, rtol: float = 1e-8) -> np.ndarray:
        """Members identical to ``truth_linear`` (the leakage guard)."""
        return np.all(np.isclose(self.res, np.asarray(truth_linear)[None, :], rtol=rtol), axis=1)

    def arrays(self, drop: Optional[np.ndarray] = None):
        """(Labels, AppRes_log, Phase) in catalog orientation, optionally
        without the members flagged in ``drop``."""
        keep = np.ones(len(self), bool) if drop is None else ~drop
        return (np.log(self.res[keep]).T, np.log(self.app_res[keep]).T, self.phase[keep].T)


@dataclass
class TestCase:
    """One synthetic test sample: the truth and its (noisy) observation."""

    __test__ = False        # not a pytest class

    index: int
    truth_linear: np.ndarray
    observation: Observation

    @property
    def truth_log10(self) -> np.ndarray:
        return np.log10(self.truth_linear)


@dataclass
class SyntheticTestSet:
    """Test models and their observations (read as stored; no noise is added here)."""

    __test__ = False

    res: np.ndarray
    app_res: np.ndarray
    phase: np.ndarray
    freqs: np.ndarray

    @classmethod
    def load(cls, data_dir: str) -> "SyntheticTestSet":
        candidates = [f"{data_dir}/kilauea_1d_freqs_{s}.txt" for s in ("test", "Test")]   # case varies
        freqs_path = next(p for p in candidates if os.path.exists(p))
        return cls(np.loadtxt(f"{data_dir}/kilauea_1d_ResModels_test.txt"),
                   np.loadtxt(f"{data_dir}/kilauea_1d_AppRes_test.txt"),
                   np.loadtxt(f"{data_dir}/kilauea_1d_Phase_test.txt"),
                   np.loadtxt(freqs_path))

    def __len__(self) -> int:
        return self.res.shape[0]

    def case(self, idx: int) -> TestCase:
        return TestCase(idx, self.res[idx], Observation.from_linear(self.freqs, self.app_res[idx], self.phase[idx]))


@dataclass
class RealStation:
    station_id: str
    path: str
    observation: Observation

    @staticmethod
    def discover(real_dir: str, pattern: str = "*_1d_response.dat") -> List[Tuple[str, str]]:
        """Sorted (station_id, path) pairs; the id is the file name minus the
        pattern's suffix (``03_08ed_1d_response.dat`` -> ``03_08ed``)."""
        suffix = pattern.split("*", 1)[1]
        paths = sorted(glob.glob(os.path.join(real_dir, pattern)))
        if not paths:
            raise FileNotFoundError(f"no files matching {pattern!r} found in {real_dir}")
        return [(os.path.basename(p)[:-len(suffix)], p) for p in paths if os.path.basename(p).endswith(suffix)]

    @classmethod
    def load(cls, station_id: str, path: str) -> "RealStation":
        """Columns: frequency, apparent resistivity, phase. Any frequency order."""
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        freqs, app_res, phase = data[:, 0], data[:, 1], data[:, 2]
        if not (np.all(np.isfinite(app_res)) and np.all(np.isfinite(phase))):
            raise ValueError(f"{path}: contains non-finite values; masked data are not supported")
        return cls(station_id, path, Observation.from_linear(freqs, app_res, phase))


@dataclass
class KilaueaDataset:
    grid: Grid
    prior: PriorEnsemble
    test: Optional[SyntheticTestSet] = None

    @classmethod
    def load(cls, data_dir: str, with_test: bool = True) -> "KilaueaDataset":
        depths = np.loadtxt(f"{data_dir}/kilauea_depth_interfaces_m.txt")
        freqs = np.loadtxt(f"{data_dir}/kilauea_1d_freqs_prior.txt")
        prior = PriorEnsemble.load(data_dir)
        test = SyntheticTestSet.load(data_dir) if with_test else None
        assert prior.res.shape[1] == depths.size - 1, "layer count does not match depth interfaces"
        if test is not None:
            assert np.array_equal(freqs, test.freqs), "prior/test frequency grids differ"
        return cls(Grid(depths, freqs), prior, test)
