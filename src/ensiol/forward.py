"""1-D magnetotelluric forward modelling and frequency-grid resampling."""
from __future__ import annotations

import cmath
from functools import partial
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d

MU_0 = 4 * np.pi * 1e-7


class SerialExecutor:
    """Drop-in for ``concurrent.futures`` executors' ``.map``, run serially.

    Experiments parallelise over test samples or stations, so inside a worker
    process the forward calls run serially.
    """

    def map(self, fn, *iterables, chunksize: int = 1):
        return list(map(fn, *iterables))


def _simulate_one(model_linear, depths, frequencies):
    return MTForward1D(depths).response(model_linear, frequencies)


class MTForward1D:
    """Forward operator for a layered 1-D earth (Wait's recursion).

    Parameters
    ----------
    depths : (n_layers + 1,) layer interfaces in metres; the last layer is a
        half-space (its thickness is never used).
    """

    def __init__(self, depths):
        self.depths = np.asarray(depths, float)
        self.thicknesses = np.diff(self.depths)

    @property
    def n_layers(self) -> int:
        return self.depths.size - 1

    @property
    def depth_centers(self) -> np.ndarray:
        return 0.5 * (self.depths[:-1] + self.depths[1:])

    def _surface_impedance_term(self, resistivity, ang_freq: float):
        res = np.flip(resistivity)
        thk = np.flip(self.thicknesses)
        q_prev = None
        for k, rho in enumerate(res):
            alpha = cmath.sqrt(1j * MU_0 * ang_freq / rho)
            if k == 0:
                q = alpha * rho
            else:
                t = cmath.tanh(alpha * thk[k - 1])
                q = (q_prev + alpha * rho * t) / (1.0 + (q_prev / (alpha * rho)) * t)
            q_prev = q
        return q_prev

    def response(self, model_linear, frequencies) -> Tuple[np.ndarray, np.ndarray]:
        """Apparent resistivity (ohm-m) and phase (deg) of one model."""
        model_linear = np.asarray(model_linear, float)
        frequencies = np.asarray(frequencies, float)
        w = 2 * np.pi * frequencies
        q = np.empty(w.size, dtype=complex)
        for i, wi in enumerate(w):
            q[i] = self._surface_impedance_term(model_linear, wi)
        app_res = (q.real ** 2 + q.imag ** 2) / (w * MU_0)
        phase = np.degrees(np.arctan2(q.imag, q.real))
        return app_res, phase

    def simulate(self, models_linear, frequencies, executor) -> Tuple[np.ndarray, np.ndarray]:
        """Responses of many models; returns arrays of shape (n_models, n_freqs)."""
        models_linear = np.asarray(models_linear, float)
        model_list = [models_linear[i] for i in range(models_linear.shape[0])]
        fn = partial(_simulate_one, depths=self.depths, frequencies=np.asarray(frequencies, float))
        results = list(executor.map(fn, model_list, chunksize=max(1, len(model_list) // 32)))
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])

    def simulate_two_grids(self, labels, catalog_freqs, obs_freqs, executor):
        """Forward-model natural-log models on the catalog grid, then resample
        to the observation grid.

        Returns (AppRes_cat_log, Phase_cat, AppRes_obs_log, Phase_obs), each of
        shape (n_freqs, n_models).
        """
        app_res_lin, phase_deg = self.simulate(np.exp(labels.T), catalog_freqs, executor)
        app_cat_log, phase_cat = np.log(app_res_lin.T), phase_deg.T
        app_obs_log, phase_obs, _, _ = match_or_interpolate(catalog_freqs, obs_freqs, app_cat_log, phase_cat)
        return app_cat_log, phase_cat, app_obs_log, phase_obs


def match_or_interpolate(freq_ens, freq_real, appres_ens, phase_ens, rtol: float = 1e-10):
    """Responses on the observation frequencies.

    If every observed frequency matches a catalog frequency, the matching rows
    are returned (``"exact"``). Otherwise log-apparent-resistivity and unwrapped
    phase are interpolated cubically in log10 frequency, extrapolating beyond
    the catalog band (``"interp"``).
    """
    freq_ens = np.asarray(freq_ens, float)
    freq_real = np.asarray(freq_real, float)
    order = np.argsort(freq_ens)
    fe_sorted = freq_ens[order]
    pos = np.searchsorted(fe_sorted, freq_real)
    hi = order[np.clip(pos, 0, freq_ens.size - 1)]
    lo = order[np.clip(pos - 1, 0, freq_ens.size - 1)]
    take = np.where(np.abs(freq_ens[hi] - freq_real) <= np.abs(freq_ens[lo] - freq_real), hi, lo)
    if np.allclose(freq_ens[take], freq_real, rtol=rtol, atol=0.0):
        return appres_ens[take, :], phase_ens[take, :], "exact", take
    log_fe, log_fr = np.log10(freq_ens), np.log10(freq_real)
    srt = np.argsort(log_fe)
    f_ar = interp1d(log_fe[srt], appres_ens[srt, :], kind="cubic", axis=0,
                    bounds_error=False, fill_value="extrapolate")
    ph_un = np.unwrap(np.deg2rad(phase_ens[srt, :]), axis=0)
    f_ph = interp1d(log_fe[srt], ph_un, kind="cubic", axis=0,
                    bounds_error=False, fill_value="extrapolate")
    return f_ar(log_fr), np.rad2deg(f_ph(log_fr)), "interp", -np.ones(freq_real.size, dtype=int)
