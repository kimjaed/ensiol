"""Before/after ensemble figure for one sounding (used for the Keller station)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._common import layered_stairs


def plot_ensemble(Labels_ens, AppRes_log_ens, Phase_ens, obs, depths_km, nrms, n_members, title,
                  out_path=None, mark_min=False, loc_res="lower right", loc_resp="upper left",
                  resp_legend_on="ar", fs=(16, 14, 11, 17), wspace=0.2, hspace_response=0.06):
    """Resistivity ensemble (left), apparent resistivity (top right) and phase (bottom right).

    Labels_ens: natural-log resistivity (n_layers, N); AppRes_log_ens: natural-log
    apparent resistivity (n_freqs, N); Phase_ens: degrees (n_freqs, N); ``obs`` is
    an :class:`~ensiol.data.Observation`; ``depths_km`` are the layer interfaces in km.
    ``mark_min`` marks the minimum of the ensemble-mean profile (half-space
    excluded) with dashed lines running to the two axes and prints its value.
    ``fs`` = (label, tick, legend, title) font sizes.
    """
    FS_LABEL, FS_TICK, FS_LEGEND, FS_TITLE = fs
    Res_lin, AppRes_lin = np.exp(Labels_ens), np.exp(AppRes_log_ens)
    periods = 1.0 / np.asarray(obs.freqs, float)

    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3], wspace=wspace, top=0.88, bottom=0.12)

    ax_res = fig.add_subplot(gs[0, 0])
    for j in range(Res_lin.shape[1]):
        v, d = layered_stairs(depths_km[:-1], Res_lin[:-1, j])
        ax_res.plot(v, d, color="0.3", alpha=0.05, lw=0.5)
    mean_stair, depth_stair = layered_stairs(depths_km[:-1], np.exp(Labels_ens[:-1].mean(axis=1)))
    ax_res.plot(mean_stair, depth_stair, color="firebrick", lw=1.8, label="Ensemble mean")
    ax_res.set_xscale("log"); ax_res.invert_yaxis()
    ax_res.set_xlabel(r"Resistivity ($\Omega\,$m)", fontsize=FS_LABEL)
    ax_res.set_ylabel("Depth (km)", fontsize=FS_LABEL)
    ax_res.tick_params(labelsize=FS_TICK)
    info = None
    if mark_min:
        mean_res = np.exp(Labels_ens[:-1].mean(axis=1))
        k = int(np.argmin(mean_res))
        rho_min, z_min = mean_res[k], 0.5 * (depths_km[k] + depths_km[k + 1])
        xl, yl = ax_res.get_xlim(), ax_res.get_ylim()
        ax_res.plot([rho_min, rho_min], [z_min, yl[0]], "--", color="k", lw=1.0, zorder=5)
        ax_res.plot([xl[0], rho_min], [z_min, z_min], "--", color="k", lw=1.0, zorder=5)
        ax_res.set_xlim(xl); ax_res.set_ylim(yl)
        info = dict(rho_min=float(rho_min), z_min_km=float(z_min), layer_km=(float(depths_km[k]), float(depths_km[k + 1])))
    ax_res.legend(fontsize=FS_LEGEND, loc=loc_res)

    gs_right = gs[0, 1].subgridspec(2, 1, hspace=hspace_response)
    ax_ar = fig.add_subplot(gs_right[0])
    for j in range(AppRes_lin.shape[1]):
        ax_ar.plot(periods, AppRes_lin[:, j], color="0.3", alpha=0.08, lw=0.6)
    ax_ar.plot(periods, AppRes_lin.mean(axis=1), color="firebrick", lw=1.6, label="Ensemble mean")
    ax_ar.plot(periods, np.exp(obs.app_res_log), "ko-", ms=4, lw=1, label="Observed")
    ax_ar.set_xscale("log"); ax_ar.set_yscale("log")
    ax_ar.set_ylabel(r"$\rho_a$ ($\Omega\,$m)", fontsize=FS_LABEL)
    ax_ar.tick_params(labelbottom=False, labelsize=FS_TICK)
    ax_ph = fig.add_subplot(gs_right[1], sharex=ax_ar)
    for j in range(Phase_ens.shape[1]):
        ax_ph.plot(periods, Phase_ens[:, j], color="0.3", alpha=0.08, lw=0.6)
    ax_ph.plot(periods, Phase_ens.mean(axis=1), color="firebrick", lw=1.6)
    ax_ph.plot(periods, obs.phase, "ko-", ms=4, lw=1)
    ax_ph.set_xscale("log")
    ax_ph.set_xlabel("Period (s)", fontsize=FS_LABEL); ax_ph.set_ylabel("Phase (deg)", fontsize=FS_LABEL)
    ax_ph.tick_params(labelsize=FS_TICK)
    handles, labels = ax_ar.get_legend_handles_labels()
    (ax_ar if resp_legend_on == "ar" else ax_ph).legend(handles, labels, fontsize=FS_LEGEND, loc=loc_resp)

    fig.suptitle(f"{title}   (nRMS = {nrms:.3f}, N = {n_members})", fontsize=FS_TITLE)
    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.mark_info = info
    return fig
