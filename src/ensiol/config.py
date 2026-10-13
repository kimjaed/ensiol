"""Configuration objects shared by every experiment.

All hyperparameters that were module-level constants in the original HPC
scripts live here as frozen dataclasses, so that one object can be passed to
every component and written verbatim into each run's ``run_config.json``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple

import numpy as np

LN10 = float(np.log(10.0))


@dataclass(frozen=True)
class NoiseModel:
    """Relative error on the impedance, expressed as noise levels on the data.

    Apparent resistivity scales with the squared impedance magnitude, so its
    log has twice the relative error; phase noise (degrees) is the angle whose
    tangent is the relative error.
    """

    rel_err: float = 0.1

    @property
    def app_res_sigma(self) -> float:
        """Noise level on natural-log apparent resistivity."""
        return 2.0 * self.rel_err

    @property
    def phase_sigma(self) -> float:
        """Noise level on phase, in degrees."""
        return float(np.degrees(np.arctan(self.rel_err)))


@dataclass(frozen=True)
class EnsIOLConfig:
    """Settings of the EnsIOL growth loop and of the EnsCGP estimate."""

    # EnsCGP / surrogate neighbourhoods
    n_neighbors: int = 500
    n_neighbors_surrogate: int = 500
    # growth loop
    n_add_max: int = 50
    max_iter: int = 100
    mean_tol: float = 1e-4
    min_accepted: int = 1
    damping_loop: float = 0.5
    growth_mode: str = "targeted"          # "targeted" or "random"
    # evaluation estimate (single backward EnsCGP update)
    damping_eval: float = 1.0
    # seeds
    base_seed: int = 0                     # per-item seed = base_seed + item index
    # minimum-distance rule for catalog members
    d_min_quantile: float = 0.5
    d_min_frac: float = 0.7
    n_probe: int = 2000
    calibration_seed: int = 12345
    # physical plausibility bounds on resistivity (ohm-m)
    res_min: float = 5e-3
    res_max: float = 5e5
    # observation noise
    noise: NoiseModel = field(default_factory=NoiseModel)

    @property
    def log_bounds(self) -> Tuple[float, float]:
        return float(np.log(self.res_min)), float(np.log(self.res_max))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ScoringConfig:
    """How accuracy and misfit are reported.

    ``exclude_halfspace`` and ``nrms_convention`` follow the manuscript:
    RMS error of the ensemble mean excludes the half-space (last layer), and
    the headline nRMS is the misfit of the mean of the members' responses
    (``"MoR"``). The other conventions are always computed and stored too.
    """

    exclude_halfspace: bool = True
    nrms_convention: str = "MoR"           # "MoR" (mean of responses) or "RoM" (response of mean)
    percentiles: Tuple[float, ...] = (2.5, 5.0, 95.0, 97.5)

    def headline_nrms_key(self) -> str:
        if self.nrms_convention not in ("MoR", "RoM"):
            raise ValueError("nrms_convention must be 'MoR' or 'RoM'")
        return f"nRMS_{self.nrms_convention}"
