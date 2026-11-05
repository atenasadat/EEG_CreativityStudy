# What changed from v7 to v8 (connectivity)

Three changes, all in Table 2 and the text around it:

1. **No notch, 100 Hz.** v7 put a 60 Hz notch on 6 recordings (all four P20 files, P11 control A2, P13 control B3) and resampled the connectivity windows to 128 Hz. v8 drops the notch everywhere. Each 4-s window is only resampled from 300 to 100 Hz. The anti-aliasing low-pass in `resample_poly` (cutoff at the new 50 Hz Nyquist) removes the 60 Hz line noise, so no separate filter is needed. No band-pass is applied to the data that go into the MVAR model.
2. **Athena's analysis is the main one again.** Table 2 now comes from her notebook cells (`wallas_analysis/wallas_mapping.ipynb`, cells 2-17, 30, 31, 32), run by `v8_her_ddtf.py` without changing the notebook: her stage alignment, all 19 people. v7 used our stage-aligned version (`23_ddtf_stagewise.py`, 14 people). Our extra two pathways (parietal to frontal) did not hold in her analysis or with session quarters, so the 19-person version is the safer claim.
3. **Beta and model order 10** (Richard's ask). Beta (13-30 Hz) is added next to theta and alpha, so 18 tests in one FDR family. The model order is 10 instead of the AIC order 5: three bands need more poles, and the residuals are closer to white (portmanteau test fails in 31% of windows at order 5, 15% at order 10). AIC (5) and BIC (2) are still computed and printed, and order 5 is reported as a check.

| Part of the paper | v7 | v8 | Script |
|---|---|---|---|
| Table 2 | Ours, N = 14, notch, 128 Hz, order 5, theta + alpha: 8 of 12, -29 to -36%, FDR .006 to .037 | Athena's, N = 19, no notch, 100 Hz, order 10, theta + alpha + beta: 8 of 18, -18 to -30%, FDR .003 to .025 | `v8_her_ddtf.py` |
| Which pathways | All 8 from parietal or occipital, incl. parietal to frontal | Parietal to occipital and occipital to parietal in all three bands, occipital to frontal in theta and alpha. Parietal to frontal FDR >= .53, beta occipital to frontal .13 | same |
| Frontal-origin pathways | FDR >= .81 | FDR >= .24 (6 tests) | same |
| Same at AIC order 5 | - | Same 8 pathways, FDR .003 to .026 | `ORDER=aic` |
| Stage-wise Wilcoxon (32 tests) | not reported | None below .05 after FDR (smallest .11) | cell 31 |
| PCA check | Ours: occipital to parietal, alpha and theta, FDR .015 | Athena's: all 6 parietal <-> occipital pathways, -26 to -34%, FDR <= .006. Frontal PCA is eye-dominated, so frontal pathways are reported from averages only | `ROI=pca` |
| Notch check | - | With the v7 notch (`NOTCH=1`) the same 8 pathways, every chi2 within 0.01 | `NOTCH=1` |
| Unstable windows | 54 of 12,648 | 61 of 13,881 | same |
| Model order | AIC 5 (39 ms), BIC 3, N = 512 | Used 10 (100 ms). AIC 5, BIC 2, N = 400 | same |
| Fig 7 (brain networks) | v06 figure (theta, alpha) | Redrawn by her cell 32, theta, alpha and beta rows | `FIG=1` |
| Fig 8 (schematic) | 2 bands | 3 bands | `v8_ddtf_schematic.py` |
| ERP cluster (F3, C3, P3, -450 to -297 ms) | p = .039 | p = .037 (no notch). Means, dz = -0.70, 13 of 15 and the held-out 13 of 15 do not change | `v7_1_erp_cluster_heldout.py` |
| After-aha power | FDR .030 | FDR .030 (the notch never touched it) | your `erp_tfr_tf_cluster_pre_event.py` |

Run (paths are examples):

```
export EEG_DATA=/path/to/cleaned CODING_CSV=/path/to/creativitycodings.csv
OUT=out/main ORDER=10 FIG=1 python v8_her_ddtf.py      # Table 2, Fig 7
OUT=out/aic ORDER=aic python v8_her_ddtf.py            # order 5 check
OUT=out/pca ORDER=10 ROI=pca python v8_her_ddtf.py     # PCA check
OUT=out/notch OUT_DIR=<folder with meeting_1013/22a_line_noise_per_file.csv> NOTCH=1 python v8_her_ddtf.py
python v8_ddtf_schematic.py out/main/tests.csv fig_ddtf_wallas_schematic.png   # Fig 8
```

Each run writes tests.csv (the 18 tests), unit.csv, coverage.csv, order_selection.csv (mean AIC and BIC for orders 1-20), posthoc.csv (stage-wise Wilcoxon) and summary.json. Reference numbers: `reference/table2_v8/`.

Our stage-aligned version (`v7_4_ddtf_runs.py`, `v7_5_ddtf_wallas_stats.py`) still runs and is kept as a check. Its no-notch 100 Hz numbers are in `reference/table2_ddtf_wallas_100hz_nonotch.json`. To get the v7 numbers back: `NOTCH=1 FS_NEW=128 python v7_4_ddtf_runs.py`.

Other code changes:
- `v7_ddtf_wrapper.py`: the notch is used only with `NOTCH=1`. `EEG_BASELINES` (optional) sets the rest-recording folder.
- `23_ddtf_stagewise.py`: default rate is 100 Hz (`FS_NEW` changes it). With `--segments wallas` it now writes the windows and stops, because its quick stats crashed when someone had only one condition in the coded stages (P11, P21, P23). The Table 2 statistics of that version were always computed by `v7_5_ddtf_wallas_stats.py`, so v7 numbers are not affected.
