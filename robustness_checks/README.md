# Robustness checks (Parsa)

New code only. The analyses that already exist in this repo (`ERP/`, `Whole Session/`, `wallas_analysis/`) are used as they are and are not duplicated here.

## Setup

Python 3.11 with MNE 1.13, numpy, scipy, pandas, statsmodels and matplotlib. Run each script from inside this folder.

Paths are in `config.py`. By default they follow your own `ERP/erp_shared.py` (`BASE_PATH`, `BASE_PATH_ICA`), so on your machine nothing needs to change. To use other folders, set `EEG_DATA`, `EEG_BASELINES`, `OUT_DIR` or `CODING_CSV`. Results go to `robustness_checks/outputs/` (not tracked by git).

## What is here

| Script | What it does | In the paper |
|---|---|---|
| `22a_line_noise_per_file.py` | 60 Hz line-noise ratio for every cleaned recording (60 Hz power vs 55-58 and 62-65 Hz, median over the 19 channels) | Methods: line-noise screening. 6 recordings are above 2.0 |
| `notch_patch.py` | Import it first, and any script that reads EEG gets a 60 Hz notch on those 6 recordings. Not used by default any more: set `NOTCH=1` to run a script with it | Check only (v7 used it) |
| `pca_roi_series.py` | PCA region signals for your Wallas notebook (Tak's request). Instructions are at the top of the file | Results: PCA robustness check |
| `01_inventory.py`, `02_counts_time.py`, `02b_aha_rate.py`, `05_fig_behaviour.py` | Aha counts, aha rate per 10 min, time on task, and the 3-panel behavior figure | Fig. 1, behavior results |
| `03_erp_cluster.py`, `04_fig_erp.py` | Participant-level ERPs, cluster test and the ERP figure | Fig. 2 |
| `v8_pre_aha_power.py`, `v8_pre_aha_power_stats.py`, `v8_fig_pre_aha_power.py` | Power before the aha vs random moments (Richard's ask), your Morlet settings | Pre-insight power results and figure |
| `csd_patch.py` | Surface Laplacian at read time (`CSD=1` in `v8_her_ddtf.py`) | Laplacian check of Table 2 (extra check, not in the paper) |
| `v8_erp_spatial.py`, `v8_fig_erp_neighbours.py` | ERP cluster test with spatial neighbours (Delaunay) and the search before the aha, cluster summary, leave-one-out, no-neighbour and notch checks, and the neighbour figure | ERP results, Fig 2 channels, appendix B |
| `v7_1_erp_cluster_heldout.py` | The same cluster test (no notch, `NOTCH=1` for the v7 version), plus the leave-one-participant-out window check (13 of 15) | ERP results, v7 |
| `07_noninsight_control.py`, `08_fig_noninsight.py`, `11_noninsight_cd_lattice19.py` | Random non-insight moments, count-matched, 1,000 draws | Fig. 3, non-insight control |
| `28_erp_no_cut_epochs.py` | ERP test without epochs that cross a removed-data cut | Robustness check |
| `v7_erp_perchannel_figs.py` | ERP at F3, C3 and P3 plotted separately, and each person's window value | Answer to Richard |
| `24_tfr_after_aha_per_person.py` | Runs your `erp_tfr_tf_cluster_pre_event.py` unchanged and gives each person's values | After-aha power, per person |
| `v8_her_ddtf.py` | Runs your connectivity cells in `wallas_mapping.ipynb` (2-17, 30, 31, 32) without changing the notebook: 100 Hz, theta + alpha + beta, order 10 (AIC and BIC still printed). `ORDER=aic`, `ROI=pca`, `NOTCH=1` give the checks, `FIG=1` draws Fig 7 with a beta row | Table 2 (8 of 18, N = 19), Fig 7, order 5, PCA and notch checks |
| `v8_ddtf_schematic.py` | The three-region schematic of Table 2, three bands | Fig 8 |
| `23_ddtf_stagewise.py`, `v7_ddtf_wrapper.py`, `v7_4_ddtf_runs.py`, `v7_5_ddtf_wallas_stats.py` | Our stage-aligned dDTF (N = 14), channel average and PCA. No notch and 100 Hz now. `NOTCH=1 FS_NEW=128` gives the v7 Table 2 | v7 Table 2, now a check |
| `v7_ddtf_schematic.py` | The v7 two-band schematic | v7 only |

`common.py` and `config.py` are helpers. `reference/13_athena_tfr_run.log` is the output of your time-frequency script, which `24` compares against. `reference/v7_table2_ddtf_wallas.json` holds the Table 2 and PCA numbers, so a rerun can be checked against it.

## Update (Nov 5): Table 2 from your cells, with beta and order 10

Also the ERP cluster test now uses spatial neighbours and the search before the aha (see `CHANGES_v7_to_v8.md`, item 4). `07`, `04` and `08` take the cluster channels from `ROI_CH`.

Table 2 now comes from your own connectivity cells (`v8_her_ddtf.py`), all 19 people, at 100 Hz with no notch. Beta (13-30 Hz) is added, so 18 tests in one FDR family, and the model order is 10 (Richard: three bands need more poles; residuals are also closer to white). AIC still picks 5 and BIC 2, and order 5 gives the same 8 pathways. Numbers and how to run each check are in `CHANGES_v7_to_v8.md`, reference outputs in `reference/table2_v8/`.

## Update (Nov 4): no notch, 100 Hz

Connectivity and ERP no longer use the 60 Hz notch. Before the MVAR fit, each 4-s window is only resampled from 300 to 100 Hz (`scipy.signal.resample_poly`). Its anti-aliasing low-pass at the new Nyquist (50 Hz) also removes the 60 Hz line noise, so no separate filter is needed. There is no band-pass on the data that go into the model. The only band-pass left is a 1-45 Hz copy used to fit the PCA weights in the PCA check, and the weights are then applied to the unfiltered data.

v7 Table 2 used the notch on 6 files and 128 Hz. To get those numbers back, run with `NOTCH=1 FS_NEW=128`. New reference numbers are in `reference/table2_ddtf_wallas_100hz_nonotch.json`, and what changed is in `CHANGES_v7_to_v8.md`.

New optional setting: `EEG_BASELINES` sets the folder with the rest recordings for the dDTF scripts (default: `BASE_PATH_ICA` in `ERP/erp_shared.py`).

## Run order

```
python 22a_line_noise_per_file.py
python 01_inventory.py && python 02_counts_time.py && python 02b_aha_rate.py && python 05_fig_behaviour.py
python 03_erp_cluster.py && python 04_fig_erp.py
python v7_1_erp_cluster_heldout.py
python 07_noninsight_control.py && python 08_fig_noninsight.py
python 24_tfr_after_aha_per_person.py --recompute
OUT=out/main ORDER=10 FIG=1 python v8_her_ddtf.py && python v8_ddtf_schematic.py out/main/tests.csv fig_ddtf_wallas_schematic.png
OUT=out/aic ORDER=aic python v8_her_ddtf.py && OUT=out/pca ROI=pca python v8_her_ddtf.py   # checks
```

## Note on filtering

MNE gives EEGLAB boundary events a duration equal to the removed length. If a filter is told to skip boundaries, it then skips real data after each cut. `notch_patch.py` sets those durations to 0 before filtering, so all data is filtered. This only matters with `NOTCH=1`.
