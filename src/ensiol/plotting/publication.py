"""Publication version of the evolution figure (IEEE two-column sizing)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..config import NoiseModel, ScoringConfig
from ._common import layered_stairs, pad_lims, pca_2d


@dataclass(frozen=True)
class PublicationStyle:
    """All styling of the figure. Sizes are in inches and points, so the figure
    can be built at its final printed size (7.16 in for a full-width figure in
    a two-column IEEE journal, 3.5 in for a single column).

    ``columns`` chooses and orders the panels per row: ``"model"``, ``"response"``,
    ``"pca_model"``, ``"pca_data"``. ``title_template`` fields: ``{it}``,
    ``{added}``, ``{rms}``, ``{nrms}``. ``legend_mode``: ``"columns"`` (one legend
    under each column), ``"row"`` (one shared legend under ``legend_row``) or ``None``.
    """

    fig_width: float = 7.16
    row_height: float = 1.55
    columns: Tuple[str, ...] = ("model", "response", "pca_model", "pca_data")
    width_ratios: Dict[str, float] = field(default_factory=lambda: dict(model=1.1, response=1.3, pca_model=1.0, pca_data=1.0))
    row_hspace: float = 0.25
    col_wspace: float = 0.35
    response_hspace: float = 0.12
    fs_row_title: float = 7.5
    fs_label: float = 7
    fs_tick: float = 6
    fs_legend: float = 6
    fs_col_title: float = 7.5
    fs_panel_label: float = 7.5
    font_family: str = "sans-serif"
    n_members: int = 500
    lw_member: float = 0.4
    alpha_member_model: float = 0.04
    alpha_member_resp: float = 0.06
    color_member: str = "0.3"
    lw_mean: float = 1.3
    color_mean: str = "firebrick"
    lw_truth: float = 1.0
    color_truth: str = "darkcyan"
    ms_obs: float = 2.0
    lw_obs: float = 0.6
    scatter_size: float = 1.2
    scatter_alpha: float = 0.3
    star_size: float = 75
    depth_unit: str = "km"
    title_template: str = "Iteration {it} ({added} added)   RMS = {rms:.2f}   nRMS = {nrms:.2f}"
    panel_labels: bool = False
    legend_mode: str = "columns"
    legend_gap_pt: float = 3
    legend_row: int = 0
    legend_gap: float = 0.006
    color_obs_star: str = "tab:orange"
    legend_ncol: int = 2
    rasterize_members: bool = True
    formats: Tuple[str, ...] = ("pdf", "png")
    dpi: int = 600
    period_axis: bool = True          # x axis of the response panels: period (s) instead of frequency (Hz)

    def but(self, **changes) -> "PublicationStyle":
        return replace(self, **changes)


def plot_evolution_publication(rows, obs, depths, ref_std, noise: NoiseModel, truth_linear,
                               style: PublicationStyle = PublicationStyle(),
                               scoring: ScoringConfig = ScoringConfig()):
    """Rows must be annotated by :meth:`Scorer.annotate_rows`."""
    S = style
    plt.rcParams.update({"font.family": S.font_family, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.minor.width": 0.4, "ytick.minor.width": 0.4})
    cols = list(S.columns)
    n_rows = len(rows)
    zscale = 1e-3 if S.depth_unit == "km" else 1.0
    zlab = f"Depth ({S.depth_unit})"
    rng = np.random.default_rng(0)
    xvals = 1.0 / obs.freqs if S.period_axis else obs.freqs
    obs_a, obs_p = obs.app_res_log, obs.phase

    a_n, p_n = noise.app_res_sigma, noise.phase_sigma
    truth_w = (np.log(truth_linear) / ref_std)[None, :]
    obs_s = np.concatenate([obs_a / a_n, obs_p / p_n])[None, :]
    model_mean, model_comp = pca_2d(np.vstack([r["Labels_est"].T / ref_std for r in rows] + [truth_w]))
    data_rows = [np.hstack([r["AppRes_real"].T / a_n, r["Phase_real"].T / p_n]) for r in rows]
    data_mean, data_comp = pca_2d(np.vstack(data_rows + [obs_s]))
    mproj = [(r["Labels_est"].T / ref_std - model_mean) @ model_comp.T for r in rows]
    dproj = [(d - data_mean) @ data_comp.T for d in data_rows]
    t_proj = (truth_w - model_mean) @ model_comp.T
    o_proj = (obs_s - data_mean) @ data_comp.T
    mxl, myl = pad_lims(mproj + [t_proj])
    dxl, dyl = pad_lims(dproj + [o_proj])
    ar_all = np.concatenate([np.exp(r["AppRes_real"]).ravel() for r in rows] + [np.exp(obs_a)])
    ph_all = np.concatenate([r["Phase_real"].ravel() for r in rows] + [obs_p])
    ar_lim, ph_lim = (ar_all.min() * 0.8, ar_all.max() * 1.2), (ph_all.min() - 5, ph_all.max() + 5)
    res_all = np.concatenate([np.exp(r["Labels_est"][:-1]).ravel() for r in rows] + [truth_linear[:-1]])
    res_lim = (res_all.min() * 0.8, res_all.max() * 1.2)

    fig = plt.figure(figsize=(S.fig_width, S.row_height * n_rows))
    gs = fig.add_gridspec(n_rows, len(cols), width_ratios=[S.width_ratios[c] for c in cols],
                          hspace=S.row_hspace, wspace=S.col_wspace)
    z_if = depths * zscale
    truth_v, z_st = layered_stairs(z_if[:-1], truth_linear[:-1])        # half-space not drawn
    raster = S.rasterize_members
    panel_axes, bottom_axes = [], {}

    def style_ax(ax):
        ax.tick_params(labelsize=S.fs_tick, length=2.5, pad=1.5)
        ax.tick_params(which="minor", length=1.5)

    for i, r in enumerate(rows):
        n_axes_before = len(fig.axes)
        first, last_row = i == 0, i == n_rows - 1
        idx = rng.choice(r["Labels_est"].shape[1], size=min(S.n_members, r["Labels_est"].shape[1]), replace=False)
        rms = r["RMS_excl"] if scoring.exclude_halfspace else r["RMS_all"]
        nrms = r["nRMS_RoM"] if scoring.nrms_convention == "RoM" else r["nRMS_MoR"]
        title = S.title_template.format(it=r["iteration"], added=r["n_added"], rms=rms, nrms=nrms)
        title_ax = None

        for j, c in enumerate(cols):
            if c == "model":
                ax = fig.add_subplot(gs[i, j]); style_ax(ax)
                Res = np.exp(r["Labels_est"])
                for m in idx:
                    v, _ = layered_stairs(z_if[:-1], Res[:-1, m])
                    ax.plot(v, z_st, color=S.color_member, alpha=S.alpha_member_model, lw=S.lw_member, rasterized=raster)
                mv, _ = layered_stairs(z_if[:-1], np.exp(r["Labels_est"].mean(axis=1))[:-1])
                ax.plot(mv, z_st, color=S.color_mean, lw=S.lw_mean, label="Ensemble mean")
                ax.plot(truth_v, z_st, "--", color=S.color_truth, lw=S.lw_truth, label="Truth")
                ax.set_xscale("log"); ax.set_xlim(res_lim); ax.set_ylim(z_if[-2], 0)
                ax.set_ylabel(zlab, fontsize=S.fs_label, labelpad=1)
                if last_row:
                    ax.set_xlabel(r"Resistivity ($\Omega\,$m)", fontsize=S.fs_label, labelpad=1)
                    bottom_axes[c] = ax
                else:
                    ax.tick_params(labelbottom=False)
                panel_axes.append(ax); title_ax = title_ax or ax

            elif c == "response":
                sub = gs[i, j].subgridspec(2, 1, hspace=S.response_hspace)
                ax1 = fig.add_subplot(sub[0]); ax2 = fig.add_subplot(sub[1], sharex=ax1)
                style_ax(ax1); style_ax(ax2)
                Ar, Ph = np.exp(r["AppRes_real"]), r["Phase_real"]
                for m in idx:
                    ax1.plot(xvals, Ar[:, m], color=S.color_member, alpha=S.alpha_member_resp, lw=S.lw_member, rasterized=raster)
                    ax2.plot(xvals, Ph[:, m], color=S.color_member, alpha=S.alpha_member_resp, lw=S.lw_member, rasterized=raster)
                ax1.plot(xvals, np.exp(r["AppRes_real"].mean(axis=1)), color=S.color_mean, lw=S.lw_mean)
                ax2.plot(xvals, Ph.mean(axis=1), color=S.color_mean, lw=S.lw_mean)
                ax1.plot(xvals, np.exp(obs_a), "o-", color="k", ms=S.ms_obs, lw=S.lw_obs, label="Observed")
                ax2.plot(xvals, obs_p, "o-", color="k", ms=S.ms_obs, lw=S.lw_obs)
                ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_ylim(ar_lim); ax2.set_ylim(ph_lim)
                ax1.set_ylabel(r"$\rho_a$ ($\Omega\,$m)", fontsize=S.fs_label, labelpad=1)
                ax2.set_ylabel(r"Phase ($^\circ$)", fontsize=S.fs_label, labelpad=1)
                ax1.tick_params(labelbottom=False)
                if last_row:
                    ax2.set_xlabel("Period (s)" if S.period_axis else "Frequency (Hz)", fontsize=S.fs_label, labelpad=1)
                    bottom_axes[c] = ax2
                else:
                    ax2.tick_params(labelbottom=False)
                panel_axes.append(ax1); title_ax = title_ax or ax1

            elif c in ("pca_model", "pca_data"):
                ax = fig.add_subplot(gs[i, j]); style_ax(ax)
                P, star, xl, yl, name, star_col = (
                    (mproj[i], t_proj, mxl, myl, "Model space", S.color_truth) if c == "pca_model"
                    else (dproj[i], o_proj, dxl, dyl, "Data space", S.color_obs_star))
                ax.scatter(P[:, 0], P[:, 1], s=S.scatter_size, color="k", alpha=S.scatter_alpha, linewidths=0, rasterized=raster)
                ax.scatter(star[:, 0], star[:, 1], marker="*", s=S.star_size, color=star_col,
                           edgecolor="k", linewidth=0.4, zorder=10)
                ax.set_xlim(xl); ax.set_ylim(yl)
                ax.set_ylabel("PC2", fontsize=S.fs_label, labelpad=1)
                if last_row:
                    ax.set_xlabel("PC1", fontsize=S.fs_label, labelpad=1)
                    bottom_axes[c] = ax
                else:
                    ax.tick_params(labelbottom=False)
                if first:
                    ax.set_title(name, fontsize=S.fs_col_title, pad=3)
                panel_axes.append(ax); title_ax = title_ax or ax

        bb = title_ax.get_position()
        fig.text(bb.x0, bb.y1 + 0.004, title, fontsize=S.fs_row_title, ha="left", va="bottom")

        if S.legend_mode == "row" and S.legend_row == i:
            row_axes = fig.axes[n_axes_before:]
            y0 = min(a.get_position().y0 for a in row_axes)
            x1 = max(a.get_position().x1 for a in row_axes)
            handles = [plt.Line2D([], [], color=S.color_mean, lw=S.lw_mean, label="Ensemble mean"),
                       plt.Line2D([], [], color=S.color_truth, lw=S.lw_truth, ls="--", label="Truth"),
                       plt.Line2D([], [], color="k", lw=S.lw_obs, marker="o", ms=S.ms_obs, label="Observed")]
            fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(x1, y0 - S.legend_gap), ncol=3,
                       fontsize=S.fs_legend, frameon=False, handlelength=1.8, columnspacing=1.2, borderaxespad=0)

    if S.legend_mode == "columns":
        mean_h = plt.Line2D([], [], color=S.color_mean, lw=S.lw_mean, label="Ensemble mean")
        star = lambda col, lab: plt.Line2D([], [], ls="none", marker="*", ms=np.sqrt(S.star_size),
                                           markerfacecolor=col, markeredgecolor="k", markeredgewidth=0.4, label=lab)
        col_handles = {
            "model": [mean_h, plt.Line2D([], [], color=S.color_truth, lw=S.lw_truth, ls="--", label="Truth")],
            "response": [mean_h, plt.Line2D([], [], color="k", lw=S.lw_obs, marker="o", ms=S.ms_obs, label="Observed")],
            "pca_model": [star(S.color_truth, "Truth")],
            "pca_data": [star(S.color_obs_star, "Observed")],
        }
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        y_low = min(to_fig.transform(ax.get_tightbbox(renderer))[0, 1] for ax in bottom_axes.values())
        y_leg = y_low - S.legend_gap_pt / 72 / fig.get_figheight()
        for c, ax in bottom_axes.items():
            bb = ax.get_position()
            fig.legend(handles=col_handles[c], loc="upper center", bbox_to_anchor=((bb.x0 + bb.x1) / 2, y_leg),
                       ncol=min(len(col_handles[c]), S.legend_ncol), fontsize=S.fs_legend, frameon=False,
                       handlelength=1.8, columnspacing=1.0, handletextpad=0.5, borderaxespad=0)

    if S.panel_labels:
        for k, ax in enumerate(panel_axes):
            ax.text(0.03, 0.97, f"({chr(97 + k)})", transform=ax.transAxes, fontsize=S.fs_panel_label,
                    ha="left", va="top", fontweight="bold", zorder=20,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8))
    return fig
