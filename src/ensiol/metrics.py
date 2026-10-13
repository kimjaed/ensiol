"""Accuracy, misfit and calibration metrics."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .catalog import Catalog
from .config import LN10, ScoringConfig
from .data import Observation
from .enscgp import Estimate, EnsCGP
from .forward import MTForward1D


def rms_log10(mean_log10: np.ndarray, truth_log10: np.ndarray, exclude_halfspace: bool = True) -> float:
    """RMS error of an (ensemble-mean) log10 resistivity profile.

    With ``exclude_halfspace`` the last layer, which has no finite thickness and
    is poorly constrained, is left out.
    """
    err = np.asarray(mean_log10) - np.asarray(truth_log10)
    if exclude_halfspace:
        err = err[:-1]
    return float(np.sqrt(np.mean(err ** 2)))


class Scorer:
    """Turns EnsCGP estimates into the reported metrics.

    Two nRMS conventions are always computed:

    * ``MoR`` (mean of responses): misfit of the mean of the members' simulated responses;
    * ``RoM`` (response of mean): misfit of the response of the ensemble-mean model.
    """

    def __init__(self, forward: MTForward1D, enscgp: EnsCGP, scoring: ScoringConfig = ScoringConfig()):
        self.forward, self.enscgp, self.scoring = forward, enscgp, scoring

    # ---- single numbers ------------------------------------------------------
    def nrms_rom(self, mean_ln: np.ndarray, obs: Observation) -> float:
        app, ph = self.forward.response(np.exp(mean_ln), obs.freqs)
        return self.enscgp.nrms(obs, np.log(app), ph)

    def score_mean_model(self, mean_ln, obs, truth_log10, nrms_mor=np.nan) -> dict:
        mean_l10 = mean_ln / LN10
        return dict(rms=rms_log10(mean_l10, truth_log10, self.scoring.exclude_halfspace),
                    nrms_rom=float(self.nrms_rom(mean_ln, obs)), nrms_mor=float(nrms_mor),
                    mean_log10=mean_l10)

    def score_catalog(self, catalog: Catalog, obs: Observation, truth_log10, seed: int, executor) -> dict:
        """Single backward EnsCGP update on ``catalog``, scored against the truth."""
        est = self.enscgp.estimate(obs, catalog, executor, seed=seed)
        return self.score_mean_model(est.Labels_est.mean(axis=1), obs, truth_log10, nrms_mor=est.nRMS)

    def annotate_rows(self, rows, obs: Observation, truth_log10) -> list:
        """Add RMS (with and without the half-space) and both nRMS conventions to
        each evolution row, in place. Used for figure titles."""
        for r in rows:
            mean_ln = r["Labels_est"].mean(axis=1)
            mean_l10 = mean_ln / LN10
            r["RMS_all"] = rms_log10(mean_l10, truth_log10, False)
            r["RMS_excl"] = rms_log10(mean_l10, truth_log10, True)
            r["nRMS_MoR"] = float(r["nRMS"])
            r["nRMS_RoM"] = float(self.nrms_rom(mean_ln, obs))
        return rows

    # ---- full per-arm summary (what the experiments store) -----------------
    def summarize(self, est: Estimate, obs: Observation, truth_log10: Optional[np.ndarray] = None) -> dict:
        pct = self.scoring.percentiles
        L10 = est.Labels_est / LN10
        A10 = est.AppRes_real / LN10
        P = est.Phase_real
        mean_ln = est.Labels_est.mean(axis=1)
        out = dict(
            nRMS_MoR=float(est.nRMS),
            nRMS_RoM=float(self.nrms_rom(mean_ln, obs)),
            res_mean_log10=L10.mean(axis=1),
            res_std_log10=L10.std(axis=1, ddof=1),
            res_pct_log10=np.percentile(L10, pct, axis=1),
            appres_mean_log10=A10.mean(axis=1),
            appres_std_log10=A10.std(axis=1, ddof=1),
            appres_pct_log10=np.percentile(A10, pct, axis=1),
            phase_mean=P.mean(axis=1),
            phase_std=P.std(axis=1, ddof=1),
            phase_pct=np.percentile(P, pct, axis=1),
            n_neighbors_used=int(est.n_neighbors_used),
        )
        if truth_log10 is not None:
            mean_l10 = L10.mean(axis=1)
            out["RMS_log10_excl"] = rms_log10(mean_l10, truth_log10, True)
            out["RMS_log10_all"] = rms_log10(mean_l10, truth_log10, False)
            out["RMS_log10"] = out["RMS_log10_excl" if self.scoring.exclude_halfspace else "RMS_log10_all"]
            out["pit"] = np.mean(L10 < truth_log10[:, None], axis=1)
        return out


def coverage_tables(S: dict, percentiles) -> tuple:
    """Coverage of the truth by the ensemble's central intervals.

    ``S`` is the stacked ``all_samples`` dictionary (arrays with a leading
    sample axis). Returns (per-layer coverage dict, rows for the summary csv).
    Intervals are taken from the stored percentiles ("quantile") and from the
    ensemble mean and standard deviation ("gaussian").
    """
    pidx = {p: i for i, p in enumerate(percentiles)}
    levels = {90: (5.0, 95.0, 1.6449), 95: (2.5, 97.5, 1.9600)}
    truth = S["truth_log10"]
    calib, rows = {}, []
    for arm in ("base", "aug"):
        pct, mu, sd = S[f"{arm}_res_pct_log10"], S[f"{arm}_res_mean_log10"], S[f"{arm}_res_std_log10"]
        for lvl, (lo, hi, z) in levels.items():
            in_q = (truth >= pct[:, pidx[lo], :]) & (truth <= pct[:, pidx[hi], :])
            in_g = np.abs(truth - mu) <= z * sd
            calib[f"{arm}_cov{lvl}_quantile_by_layer"] = in_q.mean(axis=0)
            calib[f"{arm}_cov{lvl}_gauss_by_layer"] = in_g.mean(axis=0)
            rows.append([arm, lvl, "quantile", in_q.mean(), in_q[:, :-1].mean()])
            rows.append([arm, lvl, "gaussian", in_g.mean(), in_g[:, :-1].mean()])
    return calib, rows
