"""Main-paper appendix result: whole-session coherence.

This script computes undirected coherence across electrode pairs and bands,
then saves condition-level coherence statistics and network summary figures.
"""

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wholesession_shared import (
    BANDS,
    CONDITIONS,
    ensure_dir,
    get_subject_count,
    iter_subject_raws,
    coherence_matrices,
    rs,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "wholesession_connectivity_stats")

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8"],
    "Central": ["C3", "Cz", "C4"],
    "Parietal": ["P3", "Pz", "P4", "P7", "P8"],
    "Occipital": ["O1", "O2"],
    "Temporal": ["T7", "T8"],
}

_SIGN_CACHE = {}


def roi_channel_indices(ch_names):
    return {
        roi: [ch_names.index(ch) for ch in chans if ch in ch_names]
        for roi, chans in ROIS.items()
    }


def roi_mean_value(channel_matrix, idx_a, idx_b):
    values = []
    if idx_a == idx_b:
        for pos, i in enumerate(idx_a):
            for j in idx_a[pos + 1:]:
                values.append(channel_matrix[i, j])
    else:
        for i in idx_a:
            for j in idx_b:
                values.append(channel_matrix[i, j])
    return float(np.nanmean(values)) if values else np.nan


def channel_to_roi_matrices(channel_mats, ch_names):
    roi_idx = roi_channel_indices(ch_names)
    roi_names = list(ROIS)
    roi_mats = {}
    for band, mat in channel_mats.items():
        roi_mat = np.full((len(roi_names), len(roi_names)), np.nan)
        for i, roi_a in enumerate(roi_names):
            for j, roi_b in enumerate(roi_names):
                roi_mat[i, j] = roi_mean_value(mat, roi_idx[roi_a], roi_idx[roi_b])
        roi_mats[band] = roi_mat
    return roi_mats


def condition_mean_roi_matrices(subject_mats):
    means = {cond: {} for cond in ["Control", "Treatment"]}
    for cond in means:
        for band in BANDS:
            stack = np.asarray([s[cond][band] for s in subject_mats], dtype=float)
            means[cond][band] = np.nanmean(stack, axis=0)
    return means


def plot_roi_condition_heatmaps(subject_mats):
    means = condition_mean_roi_matrices(subject_mats)
    roi_names = list(ROIS)
    fig, axes = plt.subplots(2, len(BANDS), figsize=(18, 7), dpi=300, constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#F5F5F5")

    for col_idx, band in enumerate(BANDS):
        display_mats = {}
        for cond in ["Control", "Treatment"]:
            mat = means[cond][band].copy()
            np.fill_diagonal(mat, np.nan)
            display_mats[cond] = mat
        vmax = np.nanmax([np.nanmax(display_mats["Control"]), np.nanmax(display_mats["Treatment"])])
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        last_im = None

        for row_idx, cond in enumerate(["Control", "Treatment"]):
            ax = axes[row_idx, col_idx]
            last_im = ax.imshow(display_mats[cond], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(band if row_idx == 0 else "", fontsize=12, fontweight="bold")
            ax.set_xticks(range(len(roi_names)))
            ax.set_yticks(range(len(roi_names)))
            ax.set_xticklabels(roi_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(roi_names if col_idx == 0 else [], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(CONDITIONS[cond], fontsize=12, fontweight="bold")
            ax.tick_params(length=0)

        cbar = fig.colorbar(last_im, ax=axes[:, col_idx], fraction=0.035, pad=0.01)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Coherence", fontsize=8)

    fig.savefig(os.path.join(OUTPUT_DIR, "Coherence_ROI_condition_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roi_band_profiles(subject_mats):
    roi_names = list(ROIS)
    rows = []
    for subj in subject_mats:
        for cond in ["Control", "Treatment"]:
            for band in BANDS:
                mat = subj[cond][band]
                for idx, roi in enumerate(roi_names):
                    rows.append({
                        "Pid": subj["Pid"],
                        "condition": cond,
                        "band": band,
                        "roi": roi,
                        "within_roi_coherence": mat[idx, idx],
                    })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(ROIS), figsize=(16, 3.8), dpi=300, sharey=True)
    colors = {"Control": "#0057B8", "Treatment": "#D0006F"}
    x = np.arange(len(BANDS))
    for ax, roi in zip(axes, roi_names):
        sub = df[df["roi"] == roi]
        for cond in ["Control", "Treatment"]:
            summary = (
                sub[sub["condition"] == cond]
                .groupby("band")["within_roi_coherence"]
                .agg(["mean", "sem"])
                .reindex(BANDS.keys())
                .reset_index()
            )
            ax.plot(x, summary["mean"], color=colors[cond], marker="o", label=CONDITIONS[cond])
            ax.fill_between(
                x,
                summary["mean"] - summary["sem"].fillna(0),
                summary["mean"] + summary["sem"].fillna(0),
                color=colors[cond],
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(roi, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(list(BANDS.keys()), rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Within-ROI coherence")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Coherence_ROI_band_profiles.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def exact_signs(n):
    if n not in _SIGN_CACHE:
        values = np.arange(1 << n, dtype=np.uint32)
        bits = (values[:, None] >> np.arange(n, dtype=np.uint32)) & 1
        _SIGN_CACHE[n] = np.where(bits == 1, 1.0, -1.0)
    return _SIGN_CACHE[n]


def paired_sign_flip_p_values(diff_matrix):
    diff_matrix = np.asarray(diff_matrix, dtype=float)
    if np.isfinite(diff_matrix).all():
        n = diff_matrix.shape[0]
        obs = diff_matrix.mean(axis=0)
        null = exact_signs(n) @ diff_matrix / n
        p_vals = np.mean(np.abs(null) >= (np.abs(obs)[None, :] - 1e-12), axis=0)
        return obs, p_vals, np.full(diff_matrix.shape[1], n)

    obs_vals = []
    p_vals = []
    n_vals = []
    for col in range(diff_matrix.shape[1]):
        diffs = diff_matrix[:, col]
        diffs = diffs[np.isfinite(diffs)]
        n = len(diffs)
        n_vals.append(n)
        if n == 0:
            obs_vals.append(np.nan)
            p_vals.append(np.nan)
            continue
        obs = float(diffs.mean())
        null = exact_signs(n) @ diffs / n
        obs_vals.append(obs)
        p_vals.append(float(np.mean(np.abs(null) >= (abs(obs) - 1e-12))))
    return np.asarray(obs_vals), np.asarray(p_vals), np.asarray(n_vals)


def roi_pair_stats(subject_mats):
    roi_names = list(ROIS)
    pairs = [(i, j) for i in range(len(roi_names)) for j in range(i, len(roi_names))]
    rows = []
    for band in BANDS:
        band_rows = []
        control_stack = np.asarray([s["Control"][band] for s in subject_mats], dtype=float)
        treatment_stack = np.asarray([s["Treatment"][band] for s in subject_mats], dtype=float)
        diff_matrix = np.column_stack([
            treatment_stack[:, i, j] - control_stack[:, i, j]
            for i, j in pairs
        ])
        obs_diffs, p_vals, n_vals = paired_sign_flip_p_values(diff_matrix)
        q_vals = rs.fdr_bh(p_vals)
        for pair_idx, (i, j) in enumerate(pairs):
            c = control_stack[:, i, j]
            t = treatment_stack[:, i, j]
            band_rows.append({
                "band": band,
                "roi_pair": f"{roi_names[i]}-{roi_names[j]}",
                "n": int(n_vals[pair_idx]),
                "mean_unassisted": float(np.nanmean(c)),
                "mean_llm_assisted": float(np.nanmean(t)),
                "mean_diff": float(obs_diffs[pair_idx]),
                "p": float(p_vals[pair_idx]),
                "q_fdr": float(q_vals[pair_idx]),
            })
        rows.extend(band_rows)
    return pd.DataFrame(rows)


def print_roi_stats(stats_df):
    print("\nROI coherence tests: exact paired sign-flip permutation on ROI-pair averages")
    print("FDR is applied within each band across 15 ROI pairs.")
    for band in BANDS:
        sub = stats_df[stats_df["band"] == band].sort_values("p")
        raw_n = int((sub["p"] < 0.05).sum())
        fdr_n = int((sub["q_fdr"] < 0.05).sum())
        print(f"\n{band}: raw p<0.05 ROI pairs={raw_n}, FDR q<0.05 ROI pairs={fdr_n}")
        show = sub[(sub["p"] < 0.05) | (sub["q_fdr"] < 0.05)]
        if show.empty:
            show = sub.head(5)
            print("  No raw-significant ROI pairs; showing 5 smallest p-values.")
        print(
            show[["roi_pair", "mean_unassisted", "mean_llm_assisted", "mean_diff", "p", "q_fdr"]]
            .to_string(index=False, float_format=lambda x: f"{x:.4g}")
        )


def run(max_subjects=None):
    ensure_dir(OUTPUT_DIR)
    print("\nWHOLE-SESSION CONNECTIVITY")
    print(f"Matched subjects/files: N={get_subject_count(max_subjects)}")
    print("Baseline correction first: baseline-file channel mean is subtracted from task EEG.")

    roi_subjects = []
    ch_names = None

    for subj in iter_subject_raws(max_subjects=max_subjects):
        ch_names = subj["ch_names"]
        coh_c = coherence_matrices(subj["Control"], ch_names)
        coh_t = coherence_matrices(subj["Treatment"], ch_names)
        roi_subjects.append({
            "Pid": subj["Pid"],
            "Control": channel_to_roi_matrices(coh_c, ch_names),
            "Treatment": channel_to_roi_matrices(coh_t, ch_names),
        })
        print("  coherence extracted")

    stats_df = roi_pair_stats(roi_subjects)
    print_roi_stats(stats_df)
    plot_roi_condition_heatmaps(roi_subjects)
    plot_roi_band_profiles(roi_subjects)

    print(f"\nDONE. Figures saved to: {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Whole-session regional coherence analysis.")
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(max_subjects=args.max_subjects)
