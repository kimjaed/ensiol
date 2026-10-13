"""Summary statistics and LaTeX tables for the letter.

Reads the outputs of the three experiments (``all_samples.npz`` from the synthetic run,
``results_long.csv`` from the EnsIOL-vs-MCMC run, ``station_summary.csv`` and the per-station
files from the real-data run) and writes Table I (synthetic), Table II (field) and a text
summary of every number quoted in the manuscript.

Definitions
  RMS error   root mean square over layers of (ensemble-mean log10 resistivity - true log10 resistivity),
              half-space (last layer) excluded. 10**RMS is the corresponding typical resistivity factor.
  nRMS        misfit between the observation and the MEAN OF THE ENSEMBLE'S SIMULATED RESPONSES
              (log apparent resistivity and phase, each residual divided by its noise level).
  U           factor of uncertainty: 10**(half-width of the central 95% interval of log10 resistivity),
              averaged over layers (half-space excluded) and then over samples/stations.
  Coverage    fraction of layers (half-space excluded, all samples) whose true value lies inside the
              central 95% interval of the ensemble. Needs ground truth, so synthetic only.
  Matched     for each sample the MCMC catalog is scored at the first budget of the MCMC grid that is at
              or above the number of forward calls EnsIOL used for that sample.
"""
from __future__ import annotations

import collections
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

DEPTH_BINS = ((0, 1), (1, 4), (4, np.inf))        # km
BIN_LABELS = ("0-1 km", "1-4 km", ">4 km")


def mean_med(x):
    x = np.asarray(x, float)
    return float(np.mean(x)), float(np.median(x))


def fmt2(x, nd=2):
    return f"{x[0]:.{nd}f} ({x[1]:.{nd}f})"


# ------------------------------------------------------------------------------ results
@dataclass
class SyntheticArm:
    rms: np.ndarray
    nrms: np.ndarray
    U: np.ndarray                      # (n_samples, n_layers - 1)
    cov95: float
    Ubins: Optional[List[float]] = None
    covbins: Optional[List[float]] = None


@dataclass
class SyntheticResults:
    sample_idx: list
    n_added: np.ndarray
    n_iter: np.ndarray
    exit: collections.Counter
    base: SyntheticArm
    aug: SyntheticArm

    @classmethod
    def from_npz(cls, path: str) -> "SyntheticResults":
        d = np.load(path, allow_pickle=True)
        pct = [float(p) for p in d["percentiles"]]
        i_lo, i_hi = pct.index(2.5), pct.index(97.5)
        T = d["truth_log10"]
        zc = d["depth_centers_m"][:-1] / 1e3 if "depth_centers_m" in d.files else None
        arms = {}
        for arm in ("base", "aug"):
            p = d[f"{arm}_res_pct_log10"]
            m = d[f"{arm}_res_mean_log10"]
            inside = (T[:, :-1] >= p[:, i_lo, :-1]) & (T[:, :-1] <= p[:, i_hi, :-1])
            a = SyntheticArm(rms=np.sqrt(np.mean((m[:, :-1] - T[:, :-1]) ** 2, axis=1)),
                             nrms=d[f"{arm}_nRMS_MoR"],
                             U=10 ** (0.5 * (p[:, i_hi, :-1] - p[:, i_lo, :-1])),
                             cov95=float(np.mean(inside)))
            if zc is not None:
                sel = [(zc >= lo) & (zc < hi) for lo, hi in DEPTH_BINS]
                a.Ubins = [float(a.U[:, s].mean()) for s in sel]
                a.covbins = [float(inside[:, s].mean()) for s in sel]
            arms[arm] = a
        return cls(list(d["sample_idx"]), d["n_added"].astype(float), d["n_iterations"],
                   collections.Counter(str(x) for x in d["exit_reason"]), arms["base"], arms["aug"])


@dataclass
class McmcResults:
    matched: Dict[str, np.ndarray]
    max: Dict[str, np.ndarray]
    max_budget: int

    @classmethod
    def from_csv(cls, path: str, syn: SyntheticResults) -> "McmcResults":
        R = collections.defaultdict(dict)
        for r in csv.DictReader(open(path)):
            R[(r["method"], int(r["sample_idx"]))][int(r["budget"])] = {
                k: float(r[k]) for k in ("forward_calls", "RMS_log10", "nRMS_MoR")}
        budgets = sorted({b for v in R.values() for b in v})
        idx = syn.sample_idx
        matched = [min(b for b in budgets if b >= n) for n in syn.n_added]
        out = {}
        for name, bs in (("matched", matched), ("max", [budgets[-1]] * len(idx))):
            g = lambda key: np.array([R[("mcmc_catalog", i)][b][key] for i, b in zip(idx, bs)])
            out[name] = dict(calls=g("forward_calls"), rms=g("RMS_log10"), nrms=g("nRMS_MoR"))
        return cls(out["matched"], out["max"], budgets[-1])


@dataclass
class FieldResults:
    n: int
    ids: list
    n_freqs: np.ndarray
    n_added: np.ndarray
    freq_min: np.ndarray
    freq_max: np.ndarray
    n_iter: np.ndarray
    exit: collections.Counter
    depths_km: Optional[np.ndarray]
    base: dict
    aug: dict

    @classmethod
    def from_files(cls, csv_path: str, per_station_dir: Optional[str] = None,
                   depth_file: Optional[str] = None) -> "FieldResults":
        st = list(csv.DictReader(open(csv_path)))
        f = lambda k: np.array([float(r[k]) for r in st])
        out = cls(n=len(st), ids=[r["station_id"] for r in st], n_freqs=f("n_freqs"), n_added=f("n_added"),
                  freq_min=f("freq_min"), freq_max=f("freq_max"), depths_km=None, n_iter=f("n_iterations"),
                  exit=collections.Counter(r["exit_reason"] for r in st),
                  base=dict(nrms=f("base_nRMS_MoR")), aug=dict(nrms=f("aug_nRMS_MoR")))
        if per_station_dir:
            depths = np.loadtxt(depth_file) / 1e3 if depth_file else None
            out.depths_km = depths
            for arm in (out.base, out.aug):
                arm["U"], arm["Ubins"], arm["Uprof"], arm["Rmean"] = [], [], [], []
            for sid in out.ids:
                z = np.load(os.path.join(per_station_dir, f"{sid}.npz"), allow_pickle=True)
                for name, arm in (("base", out.base), ("aug", out.aug)):
                    p = z[f"{name}_res_pct_log10"]
                    U = 10 ** (0.5 * (p[3] - p[0]))[:-1]                  # percentiles 2.5 and 97.5, half-space excluded
                    arm["U"].append(float(U.mean()))
                    arm["Uprof"].append(U)
                    arm["Rmean"].append(np.asarray(z[f"{name}_res_mean_log10"])[:-1])
                    if depths is not None:
                        c = 0.5 * (depths[:-1] + depths[1:])[: U.size]
                        arm["Ubins"].append([float(U[(c >= lo) & (c < hi)].mean()) for lo, hi in DEPTH_BINS])
            for arm in (out.base, out.aug):
                arm["U"] = np.array(arm["U"])
                arm["Ubins"] = np.array(arm["Ubins"]) if arm["Ubins"] else None
                arm["Uprof"] = np.array(arm["Uprof"])
                arm["Rmean"] = np.array(arm["Rmean"])
        return out


# ------------------------------------------------------------------------------ tables
class Table:
    def __init__(self, header: list, rows: list, colspec: str):
        self.header, self.rows, self.colspec = header, rows, colspec

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.header)
            w.writerows(self.rows)

    def write_tex(self, path: str) -> None:
        esc = lambda s: str(s).replace("%", r"\%")
        with open(path, "w") as fh:
            fh.write("\\begin{tabular}{" + self.colspec + "}\n\\toprule\n")
            fh.write(" & ".join(esc(h) for h in self.header) + " \\\\\n\\midrule\n")
            for r in self.rows:
                fh.write(" & ".join(esc(c) for c in r) + " \\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n")


class MetricsReport:
    """Builds Tables I and II and the text summary from the experiment outputs."""

    def __init__(self, synthetic: str, mcmc: Optional[str] = None, stations: Optional[str] = None,
                 per_station_dir: Optional[str] = None, depth_interfaces: Optional[str] = None,
                 prior_catalog: Optional[str] = None, out_dir: str = "./tables", echo=print):
        self.paths = dict(synthetic=synthetic, mcmc=mcmc, stations=stations, per_station_dir=per_station_dir,
                          depth_interfaces=depth_interfaces, prior_catalog=prior_catalog)
        self.out_dir, self.echo, self.log = out_dir, echo, []

    def P(self, s: str = "") -> None:
        self.echo(s)
        self.log.append(s)

    def run(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        syn = SyntheticResults.from_npz(self.paths["synthetic"])
        self.synthetic_table(syn)
        if self.paths["stations"]:
            self.field_table(FieldResults.from_files(self.paths["stations"], self.paths["per_station_dir"],
                                                     self.paths["depth_interfaces"]))
        open(f"{self.out_dir}/summary_numbers.txt", "w").write("\n".join(self.log) + "\n")
        self.echo(f"\nwrote tables and summary_numbers.txt to {self.out_dir}")

    # ---------------- Table I: synthetic
    def synthetic_table(self, syn: SyntheticResults) -> Table:
        P, b, e, n = self.P, syn.base, syn.aug, len(syn.sample_idx)
        hdr = ["Method", "Calls", "RMS error", "nRMS", "Improved (RMS/nRMS)", "U", "95% coverage"]
        rows = [["Original catalog", "0", fmt2(mean_med(b.rms)), fmt2(mean_med(b.nrms)), "--",
                 f"{b.U.mean():.2f}", f"{100 * b.cov95:.0f}%"],
                ["EnsIOL", f"{syn.n_added.mean():.0f}", fmt2(mean_med(e.rms)), fmt2(mean_med(e.nrms)),
                 f"{int((e.rms < b.rms).sum())}/{int((e.nrms < b.nrms).sum())}",
                 f"{e.U.mean():.2f}", f"{100 * e.cov95:.0f}%"]]
        P(f"SYNTHETIC ({n} samples)")
        P(f"  RMS error (log10, half-space excluded): mean {b.rms.mean():.3f} -> {e.rms.mean():.3f}; "
          f"median {np.median(b.rms):.3f} -> {np.median(e.rms):.3f}; "
          f"typical resistivity factor 10^RMS {10 ** b.rms.mean():.1f} -> {10 ** e.rms.mean():.1f}")
        P(f"  nRMS (mean of responses): mean {b.nrms.mean():.3f} -> {e.nrms.mean():.3f}; "
          f"median {np.median(b.nrms):.3f} -> {np.median(e.nrms):.3f}; "
          f"samples with nRMS <= 1.1 after: {int((e.nrms <= 1.1).sum())}, < 1: {int((e.nrms < 1).sum())}")
        P(f"  U: {b.U.mean():.2f} -> {e.U.mean():.2f};  coverage 95%: {b.cov95:.3f} -> {e.cov95:.3f}")
        if b.Ubins is not None:
            for k, lab in enumerate(BIN_LABELS):
                P(f"    {lab}: U {b.Ubins[k]:.2f} -> {e.Ubins[k]:.2f};  coverage 95% {b.covbins[k]:.2f} -> {e.covbins[k]:.2f}")
        ne = syn.n_added
        P(f"  EnsIOL calls: mean {ne.mean():.0f}, median {np.median(ne):.0f}, IQR {np.percentile(ne, [25, 75])}, "
          f"range {int(ne.min())}-{int(ne.max())}; median iterations {np.median(syn.n_iter):.0f}; exits {dict(syn.exit)}")
        if self.paths["mcmc"]:
            mc = McmcResults.from_csv(self.paths["mcmc"], syn)
            for m, label in ((mc.matched, "MCMC catalog, matched calls"), (mc.max, f"MCMC catalog, {mc.max_budget} calls")):
                rows.append([label, f"{m['calls'].mean():.0f}", fmt2(mean_med(m["rms"])), fmt2(mean_med(m["nrms"])),
                             f"{int((m['rms'] < b.rms).sum())}/{int((m['nrms'] < b.nrms).sum())}", "--", "--"])
                P(f"  {label}: mean calls {m['calls'].mean():.0f}; RMS mean {m['rms'].mean():.3f} (median {np.median(m['rms']):.3f}); "
                  f"nRMS mean {m['nrms'].mean():.3f} (median {np.median(m['nrms']):.3f}); "
                  f"EnsIOL lower RMS in {int((e.rms < m['rms']).sum())}/{n}, lower nRMS in {int((e.nrms < m['nrms']).sum())}/{n}; "
                  f"MCMC reaches EnsIOL's RMS in {int((m['rms'] <= e.rms).sum())}/{n}, its nRMS in {int((m['nrms'] <= e.nrms).sum())}/{n}")
            P(f"  call ratio ({mc.max_budget} / EnsIOL mean): {mc.max_budget / ne.mean():.1f}")
        t = Table(hdr, rows, "lccccrr")
        t.write_csv(f"{self.out_dir}/table_synthetic.csv")
        t.write_tex(f"{self.out_dir}/table_synthetic.tex")
        return t

    # ---------------- Table II: field
    def field_table(self, s: FieldResults) -> Table:
        P, b, e = self.P, s.base, s.aug
        have_U = "U" in b
        have_bins = have_U and e["Ubins"] is not None
        hdr = ["Method", "nRMS", "Stations nRMS $\\leq 1.5$"]
        hdr += ["U 0--1 km", "U 1--4 km", "U $>$4 km"] if have_bins else (["U"] if have_U else [])
        hdr += ["Models added"]
        rows = []
        for name, x in (("Original catalog", b), ("EnsIOL", e)):
            r = [name, fmt2(mean_med(x["nrms"])), f"{int((x['nrms'] <= 1.5).sum())}/{s.n}"]
            if have_bins:
                r += [f"{np.median(x['Ubins'][:, k]):.2f}" for k in range(3)]
            elif have_U:
                r.append(f"{np.median(x['U']):.2f}")
            r.append("--" if x is b else f"{np.median(s.n_added):.0f}")
            rows.append(r)
        P(f"\nFIELD ({s.n} stations, {int(s.n_freqs.min())}-{int(s.n_freqs.max())} frequencies)")
        P(f"  nRMS (mean of responses): mean {b['nrms'].mean():.3f} -> {e['nrms'].mean():.3f}; "
          f"median {np.median(b['nrms']):.3f} -> {np.median(e['nrms']):.3f}; improved {int((e['nrms'] < b['nrms']).sum())}/{s.n}; "
          f"after: <= 1.1 {int((e['nrms'] <= 1.1).sum())}, <= 1.5 {int((e['nrms'] <= 1.5).sum())}, > 2 {int((e['nrms'] > 2).sum())}, < 1 {int((e['nrms'] < 1).sum())}")
        worst = np.argsort(e["nrms"])[::-1][:5]
        P("  highest nRMS after: " + ", ".join(f"{s.ids[i]} ({e['nrms'][i]:.2f}, {int(s.n_freqs[i])} freq)" for i in worst))
        na = s.n_added
        P(f"  models added: mean {na.mean():.0f}, median {np.median(na):.0f}, IQR {np.percentile(na, [25, 75])}, "
          f"range {int(na.min())}-{int(na.max())}; median iterations {np.median(s.n_iter):.0f}; exits {dict(s.exit)}")
        P(f"  highest frequency per station: median {np.median(s.freq_max):.3g} Hz; <1 Hz at {int((s.freq_max < 1).sum())} stations, "
          f">=10 Hz at {int((s.freq_max >= 10).sum())}")
        if have_U:
            P(f"  U (layer-averaged, per station): mean {b['U'].mean():.2f} -> {e['U'].mean():.2f}; "
              f"median {np.median(b['U']):.2f} -> {np.median(e['U']):.2f}")
            if have_bins:
                for k, lab in enumerate(BIN_LABELS):
                    P(f"    U {lab}: mean {np.mean(b['Ubins'][:, k]):.2f} -> {np.mean(e['Ubins'][:, k]):.2f}; "
                      f"median {np.median(b['Ubins'][:, k]):.2f} -> {np.median(e['Ubins'][:, k]):.2f}; "
                      f"stations with larger U after: {int((e['Ubins'][:, k] > b['Ubins'][:, k]).sum())}/{s.n}")
            self._write_station_spread(s, have_bins)
            if s.depths_km is not None:
                self._write_u_profile(s)
        else:
            P("  (no --per-station-dir given: the ensemble spread U is not available for the field data)")
        t = Table(hdr, rows, "l" + "c" * (len(hdr) - 1))
        t.write_csv(f"{self.out_dir}/table_real.csv")
        t.write_tex(f"{self.out_dir}/table_real.tex")
        return t

    def _write_station_spread(self, s: FieldResults, have_bins: bool) -> None:
        b, e = s.base, s.aug
        with open(f"{self.out_dir}/station_spread.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            cols = ["station_id", "n_freqs", "freq_min", "freq_max", "n_added", "base_nRMS_MoR", "aug_nRMS_MoR", "base_U", "aug_U"]
            if have_bins:
                cols += [f"{arm}_U_{lab}" for arm in ("base", "aug") for lab in ("0-1km", "1-4km", "gt4km")]
            w.writerow(cols)
            for i, sid in enumerate(s.ids):
                r = [sid, int(s.n_freqs[i]), s.freq_min[i], s.freq_max[i], int(s.n_added[i]),
                     b["nrms"][i], e["nrms"][i], b["U"][i], e["U"][i]]
                if have_bins:
                    r += list(b["Ubins"][i]) + list(e["Ubins"][i])
                w.writerow(r)

    def _write_u_profile(self, s: FieldResults) -> None:
        b, e = s.base, s.aug
        zc = 0.5 * (s.depths_km[:-1] + s.depths_km[1:])[: b["Uprof"].shape[1]]
        pri = None
        if self.paths["prior_catalog"]:
            L = np.log10(np.loadtxt(self.paths["prior_catalog"]))[:, :-1]
            pri = (L.mean(axis=0), L.std(axis=0, ddof=1), L)
        with open(f"{self.out_dir}/U_profile_real.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["depth_km", "base_U_median", "base_U_mean", "aug_U_median", "aug_U_mean",
                        "base_logres_median", "aug_logres_median"]
                       + (["prior_logres_mean", "prior_logres_std", "base_prior_percentile", "aug_prior_percentile"] if pri else []))
            for j in range(zc.size):
                row = [zc[j], np.median(b["Uprof"][:, j]), b["Uprof"][:, j].mean(),
                       np.median(e["Uprof"][:, j]), e["Uprof"][:, j].mean(),
                       np.median(b["Rmean"][:, j]), np.median(e["Rmean"][:, j])]
                if pri:   # percentile rank of the median ensemble mean inside the prior catalog's own distribution in this layer
                    pr = lambda v: 100.0 * float(np.mean(pri[2][:, j] <= v))
                    row += [pri[0][j], pri[1][j], pr(row[5]), pr(row[6])]
                w.writerow(row)
