# What changed from Main v6 to Main v7, and which script gives each number

All v7 numbers use the same cleaned data as v6, plus a 60 Hz notch on the 6 recordings that still had line noise (`notch_patch.py`).

| Part of the paper | v6 | v7 | Script |
|---|---|---|---|
| Line noise (Methods) | Not mentioned | Every cleaned recording screened for residual 60 Hz. 6 recordings above the 2.0 threshold (all four P20 files, P11 control A2, P13 control B3) get a 60 Hz notch | `22a_line_noise_per_file.py`, `notch_patch.py` |
| Pre-insight ERP cluster | p = .037 | Same cluster (F3, C3, P3, -450 to -297 ms) with the notch. The script gives p = .0385, which rounds to .039 | `v7_1_erp_cluster_heldout.py` |
| Leave-one-participant-out check (new) | None | Window found from the other 14 people, applied to the left-out person: 13 of 15, Wilcoxon p = .0004 | `v7_1_erp_cluster_heldout.py` |
| Directed connectivity | Whole session, N = 19, 6 of 12 pathways lower | Coded Wallas stages, N = 14 (both conditions in every stage), 8 of 12 pathways lower by 29 to 36%, all from parietal or occipital cortex. New Table 2 | `v7_4_ddtf_runs.py` then `v7_5_ddtf_wallas_stats.py` (they run `23_ddtf_stagewise.py`, which follows your notebook). Reference numbers in `reference/v7_table2_ddtf_wallas.json` |
| PCA check for connectivity (new, Tak) | None | First principal component per region: occipital to parietal lower in alpha (χ² = 16.92) and theta (χ² = 16.43), FDR p = .015 | Same runs (`--roi pca`). `pca_roi_series.py` does the same swap inside your notebook |
| dDTF schematic figure (new) | None | `fig_ddtf_wallas_schematic.png` | `v7_ddtf_schematic.py` |
| After-aha power | FDR p = .030 (frontal alpha, parietal theta) | Unchanged, because the notch does not touch frequencies below 50 Hz | your `erp_tfr_tf_cluster_pre_event.py`, per person in `24_tfr_after_aha_per_person.py` |
| AI-moment posterior alpha | FDR p = .040 and .029 | Text unchanged | your notebook |
