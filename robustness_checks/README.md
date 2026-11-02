# Robustness checks (Parsa)

New code only. The analyses that already exist in this repo (`ERP/`, `Whole Session/`, `wallas_analysis/`) are used as they are and are not duplicated here.

## Setup

Python 3.11 with MNE 1.13, numpy, scipy, pandas, statsmodels and matplotlib. Run each script from inside this folder.

Paths are in `config.py`. By default they follow your own `ERP/erp_shared.py` (`BASE_PATH`, `BASE_PATH_ICA`), so on your machine nothing needs to change. To use other folders, set `EEG_DATA`, `EEG_BASELINES`, `OUT_DIR` or `CODING_CSV`. Results go to `robustness_checks/outputs/` (not tracked by git).

## What is here

| Script | What it does | In the paper |
|---|---|---|
| `22a_line_noise_per_file.py` | 60 Hz line-noise ratio for every cleaned recording (60 Hz power vs 55-58 and 62-65 Hz, median over the 19 channels) | Methods: line-noise screening. 6 recordings are above 2.0 |
| `notch_patch.py` | Import it first, and any script that reads EEG gets a 60 Hz notch on those 6 recordings | Methods: selective notch |
| `pca_roi_series.py` | PCA region signals for your Wallas notebook (Tak's request). Instructions are at the top of the file | Results: PCA robustness check |
| `01_inventory.py`, `02_counts_time.py`, `02b_aha_rate.py`, `05_fig_behaviour.py` | Aha counts, aha rate per 10 min, time on task, and the 3-panel behavior figure | Fig. 1, behavior results |
| `03_erp_cluster.py`, `04_fig_erp.py` | Participant-level ERPs, cluster test and the ERP figure | Fig. 2 |
| `v7_1_erp_cluster_heldout.py` | The same cluster test with the notch, plus the leave-one-participant-out window check (13 of 15) | ERP results, v7 |
| `07_noninsight_control.py`, `08_fig_noninsight.py`, `11_noninsight_cd_lattice19.py` | Random non-insight moments, count-matched, 1,000 draws | Fig. 3, non-insight control |
| `28_erp_no_cut_epochs.py` | ERP test without epochs that cross a removed-data cut | Robustness check |
| `v7_erp_perchannel_figs.py` | ERP at F3, C3 and P3 plotted separately, and each person's window value | Answer to Richard |
| `24_tfr_after_aha_per_person.py` | Runs your `erp_tfr_tf_cluster_pre_event.py` unchanged and gives each person's values | After-aha power, per person |
| `23_ddtf_stagewise.py`, `v7_ddtf_wrapper.py`, `v7_4_ddtf_runs.py`, `v7_5_ddtf_wallas_stats.py` | Wallas-stage dDTF with the notch, channel average and PCA, and the mixed-model test. This is the exact code behind v7 Table 2 and the PCA numbers. It follows your `wallas_mapping.ipynb` and `Whole Session/wholesession_ddtf.py` | Table 2 (8 of 12, N = 14), PCA check |
| `v7_ddtf_schematic.py` | The three-region schematic of Table 2 | dDTF schematic figure |

`common.py` and `config.py` are helpers. `reference/13_athena_tfr_run.log` is the output of your time-frequency script, which `24` compares against. `reference/v7_table2_ddtf_wallas.json` holds the Table 2 and PCA numbers, so a rerun can be checked against it.

## Run order

```
python 22a_line_noise_per_file.py
python 01_inventory.py && python 02_counts_time.py && python 02b_aha_rate.py && python 05_fig_behaviour.py
python 03_erp_cluster.py && python 04_fig_erp.py
python v7_1_erp_cluster_heldout.py
python 07_noninsight_control.py && python 08_fig_noninsight.py
python 24_tfr_after_aha_per_person.py --recompute
python v7_4_ddtf_runs.py && python v7_5_ddtf_wallas_stats.py && python v7_ddtf_schematic.py
```

## Note on filtering

MNE gives EEGLAB boundary events a duration equal to the removed length. If a filter is told to skip boundaries, it then skips real data after each cut. `notch_patch.py` sets those durations to 0 before filtering, so all data is filtered.
