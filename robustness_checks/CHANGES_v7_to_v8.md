# What changed from v7 to v8: no notch, connectivity at 100 Hz

v7 put a 60 Hz notch on 6 recordings (all four P20 files, P11 control A2, P13 control B3) and resampled the connectivity windows to 128 Hz. v8 drops the notch everywhere. Each 4-s connectivity window is only resampled from 300 to 100 Hz. The anti-aliasing low-pass in `resample_poly` (cutoff at the new 50 Hz Nyquist) removes the 60 Hz line noise, so no separate filter is needed. No band-pass is applied to the data that go into the MVAR model.

Same cleaned files, same windows, same Wallas coding, same model (order 5) and same statistics as v7.

| Part of the paper | v7 (notch, 128 Hz) | v8 (no notch, 100 Hz) | Script |
|---|---|---|---|
| Table 2, pathways lower with LLM | 8 of 12, -29 to -36%, FDR .006 to .037 | Same 8 of 12, -30 to -34%, FDR .001 to .027 | `v7_4_ddtf_runs.py`, `v7_5_ddtf_wallas_stats.py` |
| Strongest pathway | Alpha occipital to frontal, chi2 = 19.53 | Theta occipital to frontal, chi2 = 22.74, and alpha, chi2 = 22.11 | same |
| Frontal-origin pathways | FDR >= .81 | FDR >= .66 | same |
| PCA check | Occipital to parietal, alpha chi2 = 16.92 and theta 16.43, FDR .015 | Occipital to parietal, alpha chi2 = 18.91 and theta 17.64, FDR .009 (still the only 2 of 12 below .05) | same, `--roi pca` |
| Unstable windows | 54 of 12,648 | 49 of 12,648 | `23_ddtf_stagewise.py` |
| Model order | AIC 5 (39 ms), BIC 3, N = 512 samples | AIC 5 (50 ms), BIC 2, N = 400 samples | `23_ddtf_stagewise.py --aic` (BIC from the same fits) |
| ERP cluster (F3, C3, P3, -450 to -297 ms) | p = .039 | p = .037. Means, dz = -0.70, 13 of 15 and the held-out 13 of 15 do not change | `v7_1_erp_cluster_heldout.py` |
| After-aha power | FDR .030 | FDR .030 (the notch never touched it) | your `erp_tfr_tf_cluster_pre_event.py` |

Check: Table 2 at 100 Hz with the notch (`NOTCH=1`) gives the same 8 pathways. Every chi2 is within 0.01 of the no-notch value, so at 100 Hz the notch no longer matters.

New reference numbers: `reference/table2_ddtf_wallas_100hz_nonotch.json` (same layout as `reference/v7_table2_ddtf_wallas.json`).

To get the v7 numbers back: `NOTCH=1 FS_NEW=128 python v7_4_ddtf_runs.py`, and `NOTCH=1 python v7_1_erp_cluster_heldout.py`.

Other code changes:
- `v7_ddtf_wrapper.py`: the notch is used only with `NOTCH=1`. `EEG_BASELINES` (optional) sets the rest-recording folder.
- `23_ddtf_stagewise.py`: default rate is 100 Hz (`FS_NEW` changes it). With `--segments wallas` it now writes the windows and stops, because its quick stats crashed when someone had only one condition in the coded stages (P11, P21, P23). The Table 2 statistics were always computed by `v7_5_ddtf_wallas_stats.py`, so v7 numbers are not affected.
