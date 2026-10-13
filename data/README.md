# Data

Everything the figure notebooks and the cluster scripts read lives here. File names are those used throughout the project.

## `data/kilauea/`

| File | Content | In git |
|---|---|---|
| `kilauea_depth_interfaces_m.txt` | 55 layer interfaces (m); the last layer is a half-space | yes |
| `kilauea_1d_freqs_prior.txt` | 37 catalog frequencies (Hz) | yes |
| `kilauea_1d_{ResModels,AppRes,Phase}_prior.txt` | the 10,000-member prior catalog (ohm-m, ohm-m, degrees) | no, rebuilt by `scripts/build_prior_catalog.py` |
| `kilauea_1d_{ResModels,AppRes,Phase}_test.txt`, `kilauea_1d_freqs_test.txt` | the 100 synthetic test models and their noisy observations | yes (small); `kilauea_1d_freqs_test.txt` is the same grid as the prior's |

The prior catalog is not versioned because it is regenerated exactly from `data/gabbs_valley/real_data_ensemble.npy`
and the depth grid (seed 10, 30 modes, sigma 1, smoothing length 2 cells). The regenerated catalog agrees with the one
used in the manuscript to floating-point precision (relative difference of order 1e-14). For bit-exact results, deposit
the catalog files on Zenodo and download them here instead.

## `data/gabbs_valley/`

`real_data_ensemble.npy`: the Gabbs Valley station inversions (81 stations) from which the prior's mean profile
and covariance modes are derived.

## `data/kilauea_stations/`

One sounding per real station, `<station_id>_1d_response.dat`, with columns frequency (Hz), apparent resistivity
(ohm-m) and phase (degrees), in any frequency order. The Keller notebook reads `keller_1d_response.dat`. The cluster
run reads every file with at least 10 frequencies (`--min-freqs`).

## `data/new_setting_test/`

Written by `scripts/build_test_set.py` (noiseless adaptation and held-out splits and their specification). Not needed
to run the figures.
