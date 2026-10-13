#!/usr/bin/env python
"""Summary statistics and LaTeX tables (Table I, Table II) from the experiment outputs.

Inputs (all from the backward-estimate runs):
  --synthetic   all_samples.npz       from run_synthetic.py           (required)
  --mcmc        results_long.csv      from run_ensiol_vs_mcmc.py      (optional)
  --stations    station_summary.csv   from run_real_data.py           (optional)
  --per-station-dir  per_station/     from run_real_data.py           (optional, ensemble spread of the field data)
  --depth-interfaces kilauea_depth_interfaces_m.txt                   (optional, depth-binned spread)
  --prior-catalog kilauea_1d_ResModels_prior.txt                      (optional, adds the prior's per-layer statistics
                                                                       and percentile ranks to U_profile_real.csv)
Outputs in --out-dir: table_synthetic.{csv,tex}, table_real.{csv,tex}, summary_numbers.txt, and when
--per-station-dir is given station_spread.csv and, with --depth-interfaces, U_profile_real.csv.
See ensiol.summary for the definitions of RMS, nRMS, U and coverage.
"""
import argparse

from ensiol.summary import MetricsReport


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", default="outputs/synthetic/all_samples.npz")
    ap.add_argument("--mcmc")
    ap.add_argument("--stations")
    ap.add_argument("--per-station-dir")
    ap.add_argument("--depth-interfaces")
    ap.add_argument("--prior-catalog")
    ap.add_argument("--out-dir", default="outputs/tables")
    a = ap.parse_args()
    MetricsReport(a.synthetic, a.mcmc, a.stations, a.per_station_dir, a.depth_interfaces,
                  a.prior_catalog, a.out_dir).run()


if __name__ == "__main__":
    main()
