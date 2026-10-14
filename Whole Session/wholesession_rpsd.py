import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.stats as stats

from wholesession_shared import (
    BANDS,
    ensure_dir,
    finite_vlim,
    get_subject_count,
    iter_subject_raws,
    rs,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "wholesession_rpsd_stats")
CONDITION_LABELS = {
    "Baseline": "Baseline",
    "Control": "Unassisted",
    "Treatment": "LLM-assisted",
}


def relative_band_power(raw, ch_names):
    ep = raw.copy().pick(ch_names)
    spectrum = ep.compute_psd(method="welch", fmin=1.0, fmax=50.0, verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    integrate = getattr(np, "trapezoid", np.trapz)
    total = integrate(psd, freqs, axis=1)
    total = np.maximum(total, np.finfo(float).tiny)
    out = {}
    for band, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = integrate(psd[:, mask], freqs[mask], axis=1)
        out[band] = band_power / total
    return out


def shapiro_table(subject_df):
    rows = []
    for band in BANDS:
        for condition in CONDITION_LABELS:
            vals = subject_df[
                (subject_df["Band"] == band)
                & (subject_df["Condition"] == condition)
            ]["rPSD"]
            if len(vals) >= 3 and not np.all(vals == vals.iloc[0]):
                res = stats.shapiro(vals)
                rows.append({
                    "band": band,
                    "condition": CONDITION_LABELS[condition],
                    "n": len(vals),
                    "W": float(res.statistic),
                    "p": float(res.pvalue),
                })
            else:
                rows.append({
                    "band": band,
                    "condition": CONDITION_LABELS[condition],
                    "n": len(vals),
                    "W": np.nan,
                    "p": np.nan,
                })
    return pd.DataFrame(rows)


def stats_tables(subject_df):
    kruskal_rows = []
    for band in BANDS:
        groups = [
            subject_df[(subject_df["Band"] == band) & (subject_df["Condition"] == cond)]["rPSD"].to_numpy()
            for cond in CONDITION_LABELS
        ]
        res = stats.kruskal(*groups, nan_policy="omit")
        kruskal_rows.append({
            "band": band,
            "test": "Kruskal-Wallis",
            "H": float(res.statistic),
            "p": float(res.pvalue),
        })
    kruskal_df = pd.DataFrame(kruskal_rows)
    kruskal_df["q_fdr"] = rs.fdr_bh(kruskal_df["p"])

    pair_rows = []
    for band in BANDS:
        piv = subject_df[subject_df["Band"] == band].pivot(index="Pid", columns="Condition", values="rPSD")
        for a, b in [("Baseline", "Control"), ("Baseline", "Treatment"), ("Control", "Treatment")]:
            if a not in piv or b not in piv:
                continue
            test = rs.exact_wilcoxon(piv[a], piv[b])
            pair_rows.append({
                "band": band,
                "comparison": f"{CONDITION_LABELS[a]} vs {CONDITION_LABELS[b]}",
                "n": test["n"],
                "W": test["stat"],
                "p": test["p"],
                "mean_a": float(np.nanmean(piv[a])),
                "mean_b": float(np.nanmean(piv[b])),
                "mean_diff_b_minus_a": float(np.nanmean(piv[b] - piv[a])),
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df["q_fdr"] = rs.fdr_bh(pair_df["p"])
    return kruskal_df, pair_df


def plot_rpsd(channel_df, subject_df, info, ch_names):
    conditions = ["Baseline", "Control", "Treatment"]
    fig, axes = plt.subplots(len(BANDS), 4, figsize=(13.5, 3.0 * len(BANDS)), dpi=300)
    cmap = "turbo"
    violin_colors = ["#2D9CDB", "#0057B8", "#D0006F"]

    for row_idx, band in enumerate(BANDS):
        band_chan = channel_df[channel_df["Band"] == band]
        maps = []
        for condition in conditions:
            vals = (
                band_chan[band_chan["Condition"] == condition]
                .groupby("Channel")["rPSD"]
                .mean()
                .reindex(ch_names)
                .to_numpy()
            )
            maps.append(vals)
        vlim = finite_vlim(*maps)

        for col_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            picks = mne.pick_channels(info.ch_names, include=ch_names, ordered=True)
            plot_info = mne.pick_info(info, picks, copy=True)
            im, _ = mne.viz.plot_topomap(
                maps[col_idx],
                plot_info,
                axes=ax,
                show=False,
                names=ch_names,
                sensors=True,
                cmap=cmap,
                vlim=vlim,
                contours=0,
            )
            ax.set_title(CONDITION_LABELS[condition], fontsize=10)
            cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.025)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_label("rPSD", fontsize=7)

        ax = axes[row_idx, 3]
        values = [
            subject_df[(subject_df["Band"] == band) & (subject_df["Condition"] == cond)]["rPSD"].to_numpy()
            for cond in conditions
        ]
        parts = ax.violinplot(values, showmeans=False, showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=0, fontsize=8)
        ax.set_ylabel("rPSD")
        ax.set_title(f"{band} distribution", fontsize=10)
        axes[row_idx, 0].set_ylabel(band, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "rPSD_bands_topomap_violin.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(max_subjects=None):
    ensure_dir(OUTPUT_DIR)
    print("\nWHOLE-SESSION rPSD")
    print(f"Matched subjects/files: N={get_subject_count(max_subjects)}")
    print("Baseline correction first for task conditions; baseline files are not baseline-corrected.")

    channel_rows = []
    subject_rows = []
    info = None
    ch_names = None

    for subj in iter_subject_raws(max_subjects=max_subjects):
        ch_names = subj["ch_names"]
        info = subj["info"]
        for condition in ["Baseline", "Control", "Treatment"]:
            powers = relative_band_power(subj[condition], ch_names)
            for band, values in powers.items():
                subject_rows.append({
                    "Pid": subj["Pid"],
                    "Condition": condition,
                    "ConditionLabel": CONDITION_LABELS[condition],
                    "Band": band,
                    "rPSD": float(np.nanmean(values)),
                    "baseline_source": subj["baseline_source"],
                })
                for ch, value in zip(ch_names, values):
                    channel_rows.append({
                        "Pid": subj["Pid"],
                        "Condition": condition,
                        "ConditionLabel": CONDITION_LABELS[condition],
                        "Band": band,
                        "Channel": ch,
                        "rPSD": float(value),
                    })
        print("  rPSD extracted")

    subject_df = pd.DataFrame(subject_rows)
    channel_df = pd.DataFrame(channel_rows)
    subject_df.to_csv(os.path.join(OUTPUT_DIR, "subject_rpsd_band_power.csv"), index=False)
    channel_df.to_csv(os.path.join(OUTPUT_DIR, "channel_rpsd_band_power.csv"), index=False)

    normality_df = shapiro_table(subject_df)
    kruskal_df, pair_df = stats_tables(subject_df)
    normality_df.to_csv(os.path.join(OUTPUT_DIR, "rPSD_shapiro_normality.csv"), index=False)
    kruskal_df.to_csv(os.path.join(OUTPUT_DIR, "rPSD_kruskal_stats.csv"), index=False)
    pair_df.to_csv(os.path.join(OUTPUT_DIR, "rPSD_pairwise_wilcoxon_stats.csv"), index=False)

    plot_rpsd(channel_df, subject_df, info, ch_names)
    print("\nrPSD Kruskal-Wallis tests:")
    print(kruskal_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nDONE. Outputs saved to: {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Whole-session relative PSD analysis.")
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(max_subjects=args.max_subjects)
