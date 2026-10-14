"""Main-paper appendix result: whole-session spectral power.

This script computes Welch band power for full task recordings, compares
Unassisted vs LLM-assisted sessions, and saves band-power summaries/figures.
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
    WIN_SEC,
    ensure_dir,
    get_subject_count,
    iter_subject_raws,
    mean_psd_by_band,
    rs,
    windowed_psd_by_band,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "wholesession_psd_stats")


def plot_psd_time(power_time_rows):
    df = pd.DataFrame(power_time_rows)
    if df.empty:
        return
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(8, 10), dpi=300, sharex=True)
    colors = {"Control": "#0057B8", "Treatment": "#D0006F"}
    for ax, band in zip(axes, BANDS):
        sub = df[df["band"] == band]
        for cond in ["Control", "Treatment"]:
            grouped = (
                sub[sub["condition"] == cond]
                .groupby("time_s")["power_db"]
                .agg(["mean", "sem"])
                .reset_index()
            )
            ax.plot(grouped["time_s"], grouped["mean"], color=colors[cond], label=CONDITIONS[cond])
            ax.fill_between(
                grouped["time_s"],
                grouped["mean"] - grouped["sem"].fillna(0),
                grouped["mean"] + grouped["sem"].fillna(0),
                color=colors[cond],
                alpha=0.18,
                linewidth=0,
            )
        ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{band}\ndB")
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Whole-session time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "PSD_over_time_by_band.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_band_power_summary(subject_rows):
    df = pd.DataFrame(subject_rows)
    rows = []
    for _, row in df.iterrows():
        for band in BANDS:
            for cond in ["Control", "Treatment"]:
                rows.append({
                    "Pid": row["Pid"],
                    "band": band,
                    "condition": CONDITIONS[cond],
                    "power_db": row[f"{band}_{cond}_global_db"],
                })
    long = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    positions = np.arange(len(BANDS))
    width = 0.32
    colors = {"Unassisted": "#0057B8", "LLM-assisted": "#D0006F"}
    for offset, condition in [(-width / 2, "Unassisted"), (width / 2, "LLM-assisted")]:
        vals = [long[(long["band"] == band) & (long["condition"] == condition)]["power_db"].values for band in BANDS]
        bp = ax.boxplot(vals, positions=positions + offset, widths=width, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[condition])
            patch.set_alpha(0.35)
        for key in ["medians", "whiskers", "caps"]:
            for line in bp[key]:
                line.set_color("black")
    ax.set_xticks(positions)
    ax.set_xticklabels(list(BANDS.keys()))
    ax.set_ylabel("Baseline-corrected PSD (dB)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=colors["Unassisted"], lw=6, alpha=0.5),
            plt.Line2D([0], [0], color=colors["LLM-assisted"], lw=6, alpha=0.5),
        ],
        labels=["Unassisted", "LLM-assisted"],
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "PSD_band_power_conditions.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def band_power_stats(subject_rows):
    df = pd.DataFrame(subject_rows)
    rows = []
    p_vals = []
    for band in BANDS:
        c = df[f"{band}_Control_global_db"]
        t = df[f"{band}_Treatment_global_db"]
        wilc = rs.exact_wilcoxon(c, t)
        rows.append({
            "band": band,
            "n": wilc["n"],
            "mean_unassisted_db": float(np.nanmean(c)),
            "mean_llm_assisted_db": float(np.nanmean(t)),
            "mean_diff_db": float(np.nanmean(t - c)),
            "W": wilc["stat"],
            "p": wilc["p"],
        })
        p_vals.append(wilc["p"])
    q_vals = rs.fdr_bh(p_vals)
    for row, q in zip(rows, q_vals):
        row["q_fdr"] = q
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTPUT_DIR, "PSD_band_power_stats.csv"), index=False)
    print("\nPSD band condition tests:")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run(max_subjects=None):
    ensure_dir(OUTPUT_DIR)
    print("\nWHOLE-SESSION PSD")
    print(f"Matched subjects/files: N={get_subject_count(max_subjects)}")
    print("Baseline correction first: baseline-file channel mean is subtracted from task EEG.")

    subject_rows = []
    power_time_rows = []
    for subj in iter_subject_raws(max_subjects=max_subjects):
        ch_names = subj["ch_names"]
        baseline_band_power, _ = mean_psd_by_band(subj["Baseline"], ch_names)
        win_c = windowed_psd_by_band(subj["Control"], ch_names, baseline_band_power)
        win_t = windowed_psd_by_band(subj["Treatment"], ch_names, baseline_band_power)

        row = {"Pid": subj["Pid"], "baseline_source": subj["baseline_source"]}
        for band in BANDS:
            c_vals = win_c[band]
            t_vals = win_t[band]
            row[f"{band}_Control_global_db"] = float(np.nanmean(c_vals))
            row[f"{band}_Treatment_global_db"] = float(np.nanmean(t_vals))
            row[f"{band}_Diff_global_db"] = row[f"{band}_Treatment_global_db"] - row[f"{band}_Control_global_db"]
            for idx, value in enumerate(np.nanmean(c_vals, axis=1)):
                power_time_rows.append({"Pid": subj["Pid"], "condition": "Control", "band": band,
                                        "time_s": idx * WIN_SEC, "power_db": value})
            for idx, value in enumerate(np.nanmean(t_vals, axis=1)):
                power_time_rows.append({"Pid": subj["Pid"], "condition": "Treatment", "band": band,
                                        "time_s": idx * WIN_SEC, "power_db": value})
        subject_rows.append(row)
        print("  PSD extracted")

    pd.DataFrame(subject_rows).to_csv(os.path.join(OUTPUT_DIR, "subject_psd_band_power.csv"), index=False)
    plot_psd_time(power_time_rows)
    plot_band_power_summary(subject_rows)
    band_power_stats(subject_rows)
    print(f"\nDONE. Outputs saved to: {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Whole-session PSD analysis.")
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(max_subjects=args.max_subjects)
