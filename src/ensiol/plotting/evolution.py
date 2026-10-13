"""Diagnostic evolution figure (one row per stage), used by the cluster runs."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..config import LN10, NoiseModel
from ..metrics import rms_log10
from ._common import layered_stairs, pad_lims, pca_2d


def plot_evolution(rows, obs, depths, ref_std, noise: NoiseModel, out_path=None, truth_linear=None,
                   exclude_halfspace: bool = True):
    """Rows are the dictionaries of :meth:`SnapshotEvaluator.rows`.

    With ``truth_linear`` the true model is drawn and the RMS error appears in
    each row title; without it (real stations) only the data misfit is shown.
    """
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_PANEL_TITLE = 12, 11, 10, 9, 12
    ROW_HSPACE, PANEL_HSPACE = 0.35, 0.45
    has_truth = truth_linear is not None
    obs_freqs = obs.freqs
    obs_a, obs_p = obs.app_res_log, obs.phase
    if has_truth:
        truth_log10 = np.log10(truth_linear)
        for r in rows:
            mean_log10 = np.mean(r["Labels_est"], axis=1) / LN10
            r["RMS_truth"] = rms_log10(mean_log10, truth_log10, exclude_halfspace)
    n_original = rows[0]["catalog_size"]

    model_pool = np.vstack([r["Labels_est"].T / ref_std for r in rows])
    truth_w = (np.log(truth_linear) / ref_std)[None, :] if has_truth else None
    model_mean, model_comp = pca_2d(np.vstack([model_pool, truth_w]) if has_truth else model_pool)
    a_n, p_n = noise.app_res_sigma, noise.phase_sigma
    data_rows = [np.hstack([r["AppRes_real"].T / a_n, r["Phase_real"].T / p_n]) for r in rows]
    obs_scaled = np.concatenate([obs_a / a_n, obs_p / p_n])[None, :]
    data_mean, data_comp = pca_2d(np.vstack(data_rows + [obs_scaled]))
    obs_proj = (obs_scaled - data_mean) @ data_comp.T
    truth_proj = (truth_w - model_mean) @ model_comp.T if has_truth else None
    mproj = [(r["Labels_est"].T / ref_std - model_mean) @ model_comp.T for r in rows]
    dproj = [(d - data_mean) @ data_comp.T for d in data_rows]
    model_xlim, model_ylim = pad_lims(mproj + ([truth_proj] if has_truth else []))
    data_xlim, data_ylim = pad_lims(dproj + [obs_proj])
    ar_all = np.concatenate([np.exp(r["AppRes_real"]).ravel() for r in rows] + [np.exp(obs_a)])
    ph_all = np.concatenate([r["Phase_real"].ravel() for r in rows] + [obs_p])
    ar_lim, ph_lim = (ar_all.min() * 0.8, ar_all.max() * 1.2), (ph_all.min() - 5, ph_all.max() + 5)

    n_rows = len(rows)
    fig = plt.figure(figsize=(16, 3.1 * n_rows))
    gs = fig.add_gridspec(n_rows, 4, width_ratios=[1.1, 1.3, 1, 1], hspace=ROW_HSPACE, wspace=0.24)
    if has_truth:
        truth_stair, depth_stair = layered_stairs(depths[:-1], truth_linear[:-1])

    for i, r in enumerate(rows):
        n_added = r["catalog_size"] - n_original
        Labels_est = r["Labels_est"]
        Res_lin = np.exp(Labels_est)
        AR_lin, Ph = np.exp(r["AppRes_real"]), r["Phase_real"]

        ax = fig.add_subplot(gs[i, 0])
        for j in range(Res_lin.shape[1]):
            v, d = layered_stairs(depths[:-1], Res_lin[:-1, j])
            ax.plot(v, d, color="0.3", alpha=0.03, lw=0.6)
        mean_stair, d_st = layered_stairs(depths[:-1], np.exp(np.mean(Labels_est, axis=1))[:-1])
        ax.plot(mean_stair, d_st, color="tab:blue", lw=1.8, label="Ensemble mean")
        if has_truth:
            ax.plot(truth_stair, depth_stair, "r--", lw=1.6, label="Truth")
        ax.set_xscale("log"); ax.invert_yaxis()
        ax.set_ylabel("Depth (m)", fontsize=FS_LABEL); ax.tick_params(labelsize=FS_TICK)
        if i == n_rows - 1:
            ax.set_xlabel(r"Resistivity ($\Omega$m)", fontsize=FS_LABEL)
        else:
            ax.tick_params(labelbottom=False)
        if i == 0:
            ax.legend(fontsize=FS_LEGEND, loc="lower left")
        title = f"iter {r['iteration']}   ({n_added} added)   "
        title += (f"RMS={r['RMS_truth']:.3f}   " if has_truth else "") + f"nRMS={r['nRMS']:.3f}"
        ax.set_title(title, fontsize=FS_TITLE, loc="center", pad=18)

        sub = gs[i, 1].subgridspec(2, 1, hspace=PANEL_HSPACE)
        ax_ar = fig.add_subplot(sub[0])
        for j in range(AR_lin.shape[1]):
            ax_ar.plot(obs_freqs, AR_lin[:, j], color="0.3", alpha=0.08, lw=0.8)
        ax_ar.plot(obs_freqs, np.exp(np.mean(r["AppRes_real"], axis=1)), color="tab:blue", lw=1.6)
        ax_ar.plot(obs_freqs, np.exp(obs_a), "ko-", ms=4, lw=1, label="Observed")
        ax_ar.set_xscale("log"); ax_ar.set_yscale("log"); ax_ar.set_ylim(ar_lim)
        ax_ar.tick_params(labelbottom=False, labelsize=FS_TICK)
        ax_ar.set_ylabel(r"App. res.($\Omega$m)", fontsize=FS_LABEL)
        if i == 0:
            ax_ar.legend(fontsize=FS_LEGEND, loc="upper right")
        ax_ph = fig.add_subplot(sub[1], sharex=ax_ar)
        for j in range(Ph.shape[1]):
            ax_ph.plot(obs_freqs, Ph[:, j], color="0.3", alpha=0.08, lw=0.8)
        ax_ph.plot(obs_freqs, Ph.mean(axis=1), color="tab:blue", lw=1.6)
        ax_ph.plot(obs_freqs, obs_p, "ko-", ms=4, lw=1)
        ax_ph.set_xscale("log"); ax_ph.set_ylim(ph_lim); ax_ph.tick_params(labelsize=FS_TICK)
        ax_ph.set_ylabel("Phase (deg)", fontsize=FS_LABEL)
        if i == n_rows - 1:
            ax_ph.set_xlabel("Frequency (Hz)", fontsize=FS_LABEL)
        else:
            ax_ph.tick_params(labelbottom=False)

        ax_pm = fig.add_subplot(gs[i, 2])
        ax_pm.scatter(mproj[i][:, 0], mproj[i][:, 1], s=3, color="k", alpha=0.25)
        if has_truth:
            ax_pm.scatter(truth_proj[:, 0], truth_proj[:, 1], marker="*", s=140, color="red",
                          edgecolor="k", linewidth=0.5, zorder=10)
        ax_pm.set_xlim(model_xlim); ax_pm.set_ylim(model_ylim)
        ax_pm.set_xlabel("PC1", fontsize=FS_LABEL); ax_pm.set_ylabel("PC2", fontsize=FS_LABEL)
        ax_pm.tick_params(labelsize=FS_TICK)
        if i == 0:
            ax_pm.set_title("Model space", fontsize=FS_PANEL_TITLE)

        ax_pd = fig.add_subplot(gs[i, 3])
        ax_pd.scatter(dproj[i][:, 0], dproj[i][:, 1], s=3, color="k", alpha=0.25)
        ax_pd.scatter(obs_proj[:, 0], obs_proj[:, 1], marker="*", s=140,
                      color="red" if has_truth else "tab:orange", edgecolor="k", linewidth=0.5, zorder=10,
                      label=None if has_truth else "Observed")
        ax_pd.set_xlim(data_xlim); ax_pd.set_ylim(data_ylim)
        ax_pd.set_xlabel("PC1", fontsize=FS_LABEL); ax_pd.set_ylabel("PC2", fontsize=FS_LABEL)
        ax_pd.tick_params(labelsize=FS_TICK)
        if i == 0:
            ax_pd.set_title("Data space", fontsize=FS_PANEL_TITLE)
            if not has_truth:
                ax_pd.legend(fontsize=FS_LEGEND, loc="lower left")
    if out_path is not None:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return fig
