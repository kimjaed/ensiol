"""Accuracy and data misfit against the forward-model calls spent on catalog growth."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class ComparisonStyle:
    """All styling of the comparison figure. Sizes are in inches and points, so the figure
    is built at its printed size: 3.5 in for one column of a two-column IEEE journal
    (use ``layout="vertical"``), 7.16 in for full width (``layout="horizontal"``).

    ``original_style``: ``"marker"`` draws the original catalog as a grey square left of the first
    budget, ``"line"`` as a dotted horizontal line, ``None`` omits it. ``legend_mode``: ``"below"``
    (one shared legend under the figure) or ``"inside"`` (in panel ``legend_panel``). The naive grid
    sweep is the third curve (orange diamonds); ``pc_sizes`` selects which grid sizes are drawn and
    ``show_pc=False`` removes the curve.
    """

    layout: str = "horizontal"
    fig_width: float = 7.16
    panel_height: float = 2.3
    wspace: float = 0.28
    hspace: float = 0.12
    fs_label: float = 8
    fs_tick: float = 7
    fs_legend: float = 7
    fs_annot: float = 6.5
    fs_title: float = 8
    fs_panel_label: float = 8
    font_family: str = "sans-serif"
    color_iol: str = "tab:blue"
    color_mc: str = "tab:green"
    color_orig: str = "0.45"
    lw: float = 1.2
    ms: float = 3.0
    marker_iol: str = "o"
    marker_mc: str = "^"
    color_pc: str = "tab:orange"
    marker_pc: str = "D"
    show_pc: bool = True               # naive principal-component grid baseline (when given)
    pc_sizes: Optional[Tuple[int, ...]] = None    # grid sizes k to draw, e.g. (4, 5, 6, 7, 8, 10, 12, 13); None = all
    show_done_line: bool = True        # vertical line at EnsIOL's stopping point
    show_target_line: bool = True      # horizontal line at EnsIOL's final RMS (RMS panel only)
    show_nrms_one: bool = True         # nRMS = 1 reference
    lw_ref: float = 0.8
    original_style: Optional[str] = "marker"
    annotate_done: bool = False
    done_text_y: float = 0.97
    done_text_panel: int = 0
    annotate_kept: bool = False
    kept_text_pos: Tuple[float, float] = (0.98, 0.97)
    xlim: Optional[Tuple[float, float]] = None
    rms_ylim: Optional[Tuple[float, float]] = None
    nrms_ylim: Optional[Tuple[float, float]] = None
    xlabel: str = "Forward-model calls"
    rms_ylabel: str = r"RMS error ($\log_{10}\,\Omega\,$m)"
    nrms_ylabel: str = "nRMS"
    legend_mode: str = "below"
    legend_panel: int = 0
    legend_loc: str = "upper right"
    legend_ncol: Optional[int] = None  # columns of the shared legend; None = one row (use 2 for a single column)
    label_iol: str = "EnsIOL"
    label_mc: str = "MCMC"
    label_pc: str = "Grid sweep"
    label_orig: str = "Original catalog"
    titles: bool = False
    panel_labels: bool = False
    formats: Tuple[str, ...] = ("pdf", "png")
    dpi: int = 600

    def but(self, **changes) -> "ComparisonStyle":
        return replace(self, **changes)


def plot_comparison(ensiol: Dict[str, np.ndarray], mcmc: Dict[str, np.ndarray], n_iol: int,
                    n_mc_added: int, n_mc_calls: int, style: ComparisonStyle = ComparisonStyle(),
                    nrms_key: str = "nrms_mor", grid: Optional[Dict[str, np.ndarray]] = None):
    """RMS error (left) and nRMS misfit (right) against cumulative forward calls for one test case.

    ``ensiol`` and ``mcmc`` hold arrays ``calls``, ``rms``, ``nrms_mor``, ``nrms_rom`` (see
    :func:`ensiol.experiments.mcmc_comparison.cost_series`); the first entry of ``ensiol`` is the
    original catalog (0 calls), shown separately because a zero budget cannot sit on a log axis.
    Each point is one single backward EnsCGP update of that method's catalog at that stage.

    ``grid`` (optional) holds the naive grid sweep: arrays ``sizes``, ``calls``, ``rms``, ``nrms_mor``,
    ``nrms_rom``, one entry per grid size. Unlike the other two curves it is not a sequence of growing
    catalogs: every point is a separate grid, scored from the starting catalog.
    """
    S = style
    plt.rcParams.update({"font.family": S.font_family, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.minor.width": 0.4, "ytick.minor.width": 0.4})
    pos_i, pos_m = ensiol["calls"] > 0, mcmc["calls"] > 0
    iol_x, mc_x = ensiol["calls"][pos_i], mcmc["calls"][pos_m]
    iol = {"rms": ensiol["rms"][pos_i], "nrms": ensiol[nrms_key][pos_i]}
    mc = {"rms": mcmc["rms"][pos_m], "nrms": mcmc[nrms_key][pos_m]}
    orig = {"rms": float(ensiol["rms"][0]), "nrms": float(ensiol[nrms_key][0])}

    use_pc = S.show_pc and grid is not None
    if use_pc:
        sel = (np.ones(len(grid["calls"]), bool) if S.pc_sizes is None
               else np.isin(grid["sizes"], list(S.pc_sizes)))
        pc_x, pc_y = grid["calls"][sel], {"rms": grid["rms"][sel], "nrms": grid[nrms_key][sel]}
        use_pc = pc_x.size > 0

    vertical = S.layout == "vertical"
    if vertical:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(S.fig_width, 2 * S.panel_height),
                                 gridspec_kw=dict(hspace=S.hspace))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(S.fig_width, S.panel_height), gridspec_kw=dict(wspace=S.wspace))

    x_orig = min([iol_x.min(), mc_x.min()] + ([pc_x.min()] if use_pc else [])) * 0.75
    for ax, key in zip(axes, ("rms", "nrms")):
        ax.plot(iol_x, iol[key], S.marker_iol + "-", color=S.color_iol, lw=S.lw, ms=S.ms, label=S.label_iol, zorder=5)
        ax.plot(mc_x, mc[key], S.marker_mc + "-", color=S.color_mc, lw=S.lw, ms=S.ms, label=S.label_mc, zorder=4)
        if use_pc:
            ax.plot(pc_x, pc_y[key], S.marker_pc + "-", color=S.color_pc, lw=S.lw, ms=S.ms, label=S.label_pc, zorder=3)
        if S.original_style == "marker":
            ax.plot(x_orig, orig[key], "s", color=S.color_orig, ms=S.ms + 1.5, label=S.label_orig)
        elif S.original_style == "line":
            ax.axhline(orig[key], color=S.color_orig, ls=":", lw=S.lw_ref, label=S.label_orig)
        if S.show_done_line:
            ax.axvline(n_iol, color=S.color_iol, ls=":", lw=S.lw_ref, alpha=0.8)
        ax.set_xscale("log")
        ax.tick_params(labelsize=S.fs_tick, length=2.5, pad=1.5)
        ax.tick_params(which="minor", length=1.5)
        if S.xlim is not None:
            ax.set_xlim(S.xlim)

    ax_r, ax_n = axes
    if S.show_target_line:
        ax_r.axhline(iol["rms"][-1], color=S.color_iol, ls=":", lw=S.lw_ref, alpha=0.8)
    if S.show_nrms_one:
        ax_n.axhline(1.0, color="k", ls="--", lw=S.lw_ref)
    if S.rms_ylim is not None:
        ax_r.set_ylim(S.rms_ylim)
    if S.nrms_ylim is not None:
        ax_n.set_ylim(S.nrms_ylim)
    ax_r.set_ylabel(S.rms_ylabel, fontsize=S.fs_label, labelpad=2)
    ax_n.set_ylabel(S.nrms_ylabel, fontsize=S.fs_label, labelpad=2)
    for ax in (axes[-1:] if vertical else axes):
        ax.set_xlabel(S.xlabel, fontsize=S.fs_label, labelpad=2)
    if S.titles:
        ax_r.set_title("Accuracy vs. cost", fontsize=S.fs_title, pad=3)
        ax_n.set_title("Data misfit vs. cost", fontsize=S.fs_title, pad=3)
    if S.annotate_done:
        ax_d = axes[S.done_text_panel]
        ax_d.text(n_iol * 1.12, S.done_text_y, f"{S.label_iol} stops\n({n_iol:,} calls)",
                  transform=ax_d.get_xaxis_transform(), fontsize=S.fs_annot, color=S.color_iol, va="top", ha="left")
    if S.annotate_kept:
        ax_n.text(*S.kept_text_pos, f"{S.label_mc}: {n_mc_added:,} of {n_mc_calls:,}\ncalls added to catalog",
                  transform=ax_n.transAxes, fontsize=S.fs_annot, color=S.color_mc, ha="right", va="top")

    handles = [Line2D([], [], color=S.color_iol, marker=S.marker_iol, lw=S.lw, ms=S.ms, label=S.label_iol),
               Line2D([], [], color=S.color_mc, marker=S.marker_mc, lw=S.lw, ms=S.ms, label=S.label_mc)]
    if use_pc:
        handles.append(Line2D([], [], color=S.color_pc, marker=S.marker_pc, lw=S.lw, ms=S.ms, label=S.label_pc))
    if S.original_style == "marker":
        handles.append(Line2D([], [], color=S.color_orig, marker="s", ls="none", ms=S.ms + 1.5, label=S.label_orig))
    elif S.original_style == "line":
        handles.append(Line2D([], [], color=S.color_orig, ls=":", lw=S.lw_ref, label=S.label_orig))
    if S.legend_mode == "inside":
        axes[S.legend_panel].legend(handles=handles, loc=S.legend_loc, fontsize=S.fs_legend,
                                    frameon=True, framealpha=0.85, edgecolor="none")
    elif S.legend_mode == "below":
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        y_low = min(to_fig.transform(ax.get_tightbbox(renderer))[0, 1] for ax in axes)
        fig.legend(handles=handles, loc="upper center", ncol=S.legend_ncol or len(handles), fontsize=S.fs_legend,
                   bbox_to_anchor=(0.5, y_low - 2 / 72 / fig.get_figheight()), frameon=False,
                   borderaxespad=0, handlelength=2.0, columnspacing=1.5)
    if S.panel_labels:
        for k, ax in enumerate(axes):
            ax.text(0.02, 0.98, f"({chr(97 + k)})", transform=ax.transAxes, fontsize=S.fs_panel_label,
                    fontweight="bold", ha="left", va="top")
    return fig


def accuracy_vs_cost_summary(budgets: np.ndarray, series: Dict[str, Dict[str, np.ndarray]],
                             labels: Dict[str, str], colors: Dict[str, str], median_iol_total: float,
                             n_samples: int, nrms_key: str = "nrms_rom"):
    """Median and interquartile band over test samples, per method.

    ``series[method][metric]`` has shape (n_samples, n_budgets); budget 0 (the
    original catalog) is drawn as a grey square at the left edge.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    pos = budgets > 0
    x = budgets[pos]
    first = next(iter(series))
    for ax, key in zip(axes, ("rms", nrms_key)):
        ax.plot(x[0] * 0.8, np.median(series[first][key][:, 0]), "s", color="0.4", ms=5, label="Original catalog")
        for m, s in series.items():
            v = s[key][:, pos]
            ax.plot(x, np.nanmedian(v, axis=0), "o-", color=colors[m], label=labels[m], ms=4)
            ax.fill_between(x, np.nanpercentile(v, 25, axis=0), np.nanpercentile(v, 75, axis=0),
                            color=colors[m], alpha=0.15)
        ax.set_xscale("log"); ax.axvline(median_iol_total, color="C0", ls=":", lw=1)
        ax.set_xlabel("Forward-model calls (budget)"); ax.legend(fontsize=8)
    axes[0].set_ylabel("RMS error (log$_{10}$ Ω·m)")
    axes[1].set_ylabel("nRMS"); axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    fig.suptitle(f"Median and interquartile range over {n_samples} test samples "
                 f"(dotted: median EnsIOL total)", fontsize=10)
    return fig
