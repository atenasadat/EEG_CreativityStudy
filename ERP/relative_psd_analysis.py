"""Main-paper appendix result: task engagement relative to rest.

This script computes relative PSD across baseline, Unassisted, and LLM-assisted
recordings, runs band-wise tests, and saves the rPSD topomap/violin figure.
"""

import os
import warnings

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

warnings.filterwarnings("ignore", message="Unknown types found*")
warnings.filterwarnings("ignore", message="The data contains 'boundary' events*")
warnings.filterwarnings("ignore", message="Limited .* annotation.*")
warnings.filterwarnings("ignore", message="Concatenation of Annotations.*")
warnings.filterwarnings("ignore", message="Data file name in EEG.data .*")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from erp_shared import (
    BASE_PATH,
    BASE_PATH_ICA,
    STANDARD_CHANNELS,
    get_trigger_file_groups,
    load_aha_epochs_from_files,
    load_resting_baseline_epochs,
)


mne.set_log_level("ERROR")

BANDS = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 13.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 50.0),
]

CONDITIONS = ["Baseline", "Unassisted", "LLM-assisted"]
EPOCH_WINDOW = (-5.0, 5.0)
RESTING_BASELINE_EPOCH_DURATION = 10.0
PSD_RANGE = (1.0, 50.0)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "relative_psd_outputs")
FIGURE_PATH = os.path.join(OUTPUT_DIR, "relative_psd_bands_topomap_violin.png")
STATS_PATH = os.path.join(OUTPUT_DIR, "relative_psd_band_stats.csv")


def fdr_bh(p_values):
    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any():
        return out

    pv = p[good]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(pv)
    adjusted[order] = np.clip(ranked, 0, 1)
    out[good] = adjusted
    return out


def p_stars(p_value):
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def compute_relative_band_power(epochs, channels):
    ep = epochs.copy().pick(channels)
    spectrum = ep.compute_psd(
        method="welch",
        fmin=PSD_RANGE[0],
        fmax=PSD_RANGE[1],
        verbose=False,
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    integrate = getattr(np, "trapezoid", np.trapz)
    total = integrate(psd, freqs, axis=2)
    total = np.maximum(total, np.finfo(float).tiny)

    band_values = {}
    for band_name, fmin, fmax in BANDS:
        mask = (freqs >= fmin) & (freqs <= fmax)
        if mask.sum() < 2:
            band_values[band_name] = np.full(len(ep.ch_names), np.nan)
            continue

        band_power = integrate(psd[:, :, mask], freqs[mask], axis=2)
        rel_power = band_power / total
        band_values[band_name] = np.nanmean(rel_power, axis=0)

    return band_values, ep.info, ep.ch_names


def baseline_correct_rpsd(values, baseline_values):
    tiny = np.finfo(float).tiny
    return 10.0 * np.log10(
        np.maximum(values, tiny) / np.maximum(baseline_values, tiny)
    )


def build_relative_psd_table():
    matched_pids, c_groups, t_groups = get_trigger_file_groups(BASE_PATH, verbose=False)
    rows = []
    subject_rows = []
    info = None
    ch_names = None
    skipped_missing_baseline = []
    trial_counts = []

    print(f"matched_subjects={len(matched_pids)}")

    for idx, pid in enumerate(matched_pids, start=1):
        print(f"processing_subject={idx}/{len(matched_pids)} pid={pid}", flush=True)
        ep_control = load_aha_epochs_from_files(
            c_groups[pid],
            base_path=BASE_PATH,
            tmin=EPOCH_WINDOW[0],
            tmax=EPOCH_WINDOW[1],
            baseline=None,
        )
        ep_treatment = load_aha_epochs_from_files(
            t_groups[pid],
            base_path=BASE_PATH,
            tmin=EPOCH_WINDOW[0],
            tmax=EPOCH_WINDOW[1],
            baseline=None,
        )
        ep_baseline, baseline_file = load_resting_baseline_epochs(
            pid,
            base_path_ica=BASE_PATH_ICA,
            channels=STANDARD_CHANNELS,
            epoch_duration=RESTING_BASELINE_EPOCH_DURATION,
        )

        if ep_control is None or ep_treatment is None:
            continue
        if ep_baseline is None or not len(ep_baseline):
            skipped_missing_baseline.append(pid)
            print(f"skipping_subject={pid} reason=missing_baseline_file")
            continue

        condition_epochs = {
            "Baseline": ep_baseline,
            "Unassisted": ep_control,
            "LLM-assisted": ep_treatment,
        }
        trial_counts.append({
            "Pid": pid,
            "Baseline": len(ep_baseline),
            "Unassisted": len(ep_control),
            "LLM-assisted": len(ep_treatment),
            "baseline_source": baseline_file,
        })

        baseline_band_values, this_info, this_ch_names = compute_relative_band_power(
            ep_baseline,
            STANDARD_CHANNELS,
        )
        if info is None:
            info = this_info
            ch_names = this_ch_names

        for condition, epochs in condition_epochs.items():
            band_values, _, this_ch_names = compute_relative_band_power(
                epochs,
                STANDARD_CHANNELS,
            )

            for band_name, _, _ in BANDS:
                raw_values = band_values[band_name]
                if condition == "Baseline":
                    corrected_values = np.full_like(raw_values, np.nan)
                else:
                    corrected_values = baseline_correct_rpsd(
                        raw_values,
                        baseline_band_values[band_name],
                    )
                corrected_mean = (
                    float(np.nanmean(corrected_values))
                    if np.isfinite(corrected_values).any()
                    else np.nan
                )
                subject_rows.append({
                    "Pid": pid,
                    "Condition": condition,
                    "Band": band_name,
                    "rPSD": float(np.nanmean(raw_values)),
                    "rPSD_db_from_baseline": corrected_mean,
                })
                for ch_name, raw_value, corrected_value in zip(
                    this_ch_names,
                    raw_values,
                    corrected_values,
                ):
                    rows.append({
                        "Pid": pid,
                        "Condition": condition,
                        "Band": band_name,
                        "Channel": ch_name,
                        "rPSD": float(raw_value),
                        "rPSD_db_from_baseline": float(corrected_value),
                    })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(subject_rows),
        pd.DataFrame(trial_counts),
        info,
        ch_names,
        skipped_missing_baseline,
    )


def run_stats(subject_df):
    rows = []
    pairwise = []
    normality = []

    for band_name, _, _ in BANDS:
        wide = (
            subject_df[subject_df["Band"] == band_name]
            .pivot(index="Pid", columns="Condition", values="rPSD")
            .dropna(subset=CONDITIONS)
        )
        for condition in CONDITIONS:
            values = wide[condition].dropna().to_numpy()
            if len(values) >= 3:
                shapiro = stats.shapiro(values)
                statistic = float(shapiro.statistic)
                p_value = float(shapiro.pvalue)
            else:
                statistic = np.nan
                p_value = np.nan
            normality.append({
                "measure": "raw_rPSD",
                "band": band_name,
                "test": "Shapiro-Wilk",
                "comparison": condition,
                "n_subjects": len(values),
                "statistic": statistic,
                "p_uncorrected": p_value,
            })

        omnibus = stats.kruskal(
            wide["Baseline"],
            wide["Unassisted"],
            wide["LLM-assisted"],
        )
        rows.append({
            "measure": "raw_rPSD",
            "band": band_name,
            "test": "Kruskal-Wallis",
            "comparison": "Baseline vs Unassisted vs LLM-assisted",
            "n_subjects": len(wide),
            "statistic": float(omnibus.statistic),
            "p_uncorrected": float(omnibus.pvalue),
        })

        for left, right in [
            ("Baseline", "Unassisted"),
            ("Baseline", "LLM-assisted"),
            ("Unassisted", "LLM-assisted"),
        ]:
            res = stats.wilcoxon(
                wide[right],
                wide[left],
                zero_method="wilcox",
                method="auto",
            )
            pairwise.append({
                "measure": "raw_rPSD",
                "band": band_name,
                "test": "Wilcoxon signed-rank",
                "comparison": f"{left} vs {right}",
                "n_subjects": len(wide),
                "statistic": float(res.statistic),
                "p_uncorrected": float(res.pvalue),
            })

        wide_corrected = (
            subject_df[subject_df["Band"] == band_name]
            .pivot(index="Pid", columns="Condition", values="rPSD_db_from_baseline")
            .dropna(subset=["Unassisted", "LLM-assisted"])
        )
        res = stats.wilcoxon(
            wide_corrected["LLM-assisted"],
            wide_corrected["Unassisted"],
            zero_method="wilcox",
            method="auto",
        )
        pairwise.append({
            "measure": "baseline_corrected_task_rPSD_dB",
            "band": band_name,
            "test": "Wilcoxon signed-rank",
            "comparison": "Unassisted vs LLM-assisted",
            "n_subjects": len(wide_corrected),
            "statistic": float(res.statistic),
            "p_uncorrected": float(res.pvalue),
        })

    pairwise_df = pd.DataFrame(pairwise)
    pairwise_df["p_fdr"] = np.nan
    for _, index in pairwise_df.groupby("measure").groups.items():
        pairwise_df.loc[index, "p_fdr"] = fdr_bh(pairwise_df.loc[index, "p_uncorrected"])
    normality_df = pd.DataFrame(normality)
    normality_df["p_fdr"] = fdr_bh(normality_df["p_uncorrected"])
    omnibus_df = pd.DataFrame(rows)
    omnibus_df["p_fdr"] = fdr_bh(omnibus_df["p_uncorrected"])
    return pd.concat([normality_df, omnibus_df, pairwise_df], ignore_index=True)


def add_sig_bar(ax, x1, x2, y, text, height):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2, y + height, text, ha="center", va="bottom", fontsize=10)


def plot_results(channel_df, subject_df, stats_df, info, ch_names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    significant_omnibus_bands = set(
        stats_df[
            (stats_df["measure"] == "raw_rPSD")
            & (stats_df["test"] == "Kruskal-Wallis")
            & (stats_df["p_fdr"] < 0.05)
        ]["band"]
    )
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    sns.set_theme(
        style="white",
        context="paper",
        font_scale=1.2,
        rc={"font.family": "Times New Roman", "mathtext.fontset": "stix"},
    )

    fig = plt.figure(figsize=(20, 21), dpi=300)
    grid = fig.add_gridspec(
        len(BANDS),
        5,
        width_ratios=[1.35, 1.35, 1.35, 0.08, 1.50],
        left=0.035,
        right=0.985,
        top=0.955,
        bottom=0.035,
        hspace=0.34,
        wspace=0.38,
    )

    letters = ["A", "B", "C", "D", "E"]
    palette = {
        "Baseline": "#79D200",
        "Unassisted": "#2E86DE",
        "LLM-assisted": "#B95AD9",
    }
    for row, (band_name, fmin, fmax) in enumerate(BANDS):
        band_channel = channel_df[channel_df["Band"] == band_name]
        topomap_values = []
        for condition in CONDITIONS:
            means = (
                band_channel[band_channel["Condition"] == condition]
                .groupby("Channel")["rPSD"]
                .mean()
            )
            topomap_values.append(np.asarray([means.get(ch, np.nan) for ch in ch_names]))

        finite_values = np.concatenate([values[np.isfinite(values)] for values in topomap_values])
        vmin, vmax = np.nanmin(finite_values), np.nanmax(finite_values)

        for col, (condition, values) in enumerate(zip(CONDITIONS, topomap_values)):
            ax = fig.add_subplot(grid[row, col])
            im, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                names=ch_names,
                sensors=True,
                contours=6,
                cmap="RdBu_r",
                vlim=(vmin, vmax),
            )
            ax.set_title(condition, fontsize=18, pad=5)
            if col == 0:
                row_label_y = min(ax.get_position().y1 + 0.018, 0.982)
                fig.text(
                    0.014,
                    row_label_y,
                    letters[row],
                    fontsize=18,
                    fontweight="bold",
                    ha="left",
                    va="center",
                )
                fig.text(
                    0.032,
                    row_label_y,
                    f"{band_name} Band [{fmin:.0f}-{fmax:.0f} Hz]",
                    fontsize=18,
                    fontweight="bold",
                    ha="left",
                    va="center",
                )

        cax = fig.add_subplot(grid[row, 3])
        cax_pos = cax.get_position()
        cax.set_position([
            cax_pos.x0 - 0.012,
            cax_pos.y0,
            cax_pos.width,
            cax_pos.height,
        ])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([vmin, (vmin + vmax) / 2.0, vmax])
        cbar.ax.set_title("rPSD", fontsize=16, pad=4)
        cbar.ax.tick_params(labelsize=12)

        ax = fig.add_subplot(grid[row, 4])
        band_subject = subject_df[subject_df["Band"] == band_name]
        sns.violinplot(
            data=band_subject,
            x="Condition",
            y="rPSD",
            hue="Condition",
            order=CONDITIONS,
            hue_order=CONDITIONS,
            palette=palette,
            inner="box",
            cut=0,
            linewidth=0.8,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=band_subject,
            x="Condition",
            y="rPSD",
            order=CONDITIONS,
            color="black",
            size=2.5,
            alpha=0.45,
            jitter=0.12,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("rPSD", fontsize=14)
        ax.tick_params(axis="x", labelrotation=0, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

        band_stats = stats_df[
            (stats_df["band"] == band_name)
            & (stats_df["test"] == "Wilcoxon signed-rank")
            & (stats_df["measure"] == "raw_rPSD")
            & (stats_df["band"].isin(significant_omnibus_bands))
            & (stats_df["p_fdr"] < 0.05)
        ]
        y_min, y_max = ax.get_ylim()
        step = (y_max - y_min) * 0.08
        start = y_max + step * 0.2
        pair_to_x = {
            "Baseline vs Unassisted": (0, 1),
            "Baseline vs LLM-assisted": (0, 2),
            "Unassisted vs LLM-assisted": (1, 2),
        }
        for offset, (_, stat_row) in enumerate(band_stats.iterrows()):
            x1, x2 = pair_to_x[stat_row["comparison"]]
            label = (
                f"{p_stars(stat_row['p_fdr'])}  Wilcoxon $W$ = {stat_row['statistic']:.0f}, "
                f"$p_{{FDR}}$ = {stat_row['p_fdr']:.4f}"
            )
            add_sig_bar(ax, x1, x2, start + offset * step * 1.6, label, step * 0.35)
        ax.set_ylim(y_min, start + step * 4.6)

    fig.text(
        0.985,
        0.012,
        "Omnibus: Kruskal-Wallis $H$ (FDR across bands). Post hoc: Wilcoxon signed-rank $W$ "
        "(FDR across pairs and bands), shown only for bands with a significant omnibus.",
        ha="right",
        va="bottom",
        fontsize=12,
        style="italic",
    )
    fig.add_artist(Rectangle(
        (0.005, 0.005),
        0.99,
        0.99,
        transform=fig.transFigure,
        fill=False,
        edgecolor="black",
        linewidth=1.2,
        clip_on=False,
    ))
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_figure={FIGURE_PATH}")


def main():
    channel_df, subject_df, trial_counts, info, ch_names, skipped_missing_baseline = build_relative_psd_table()
    if channel_df.empty:
        raise RuntimeError("No relative PSD data were computed.")

    stats_df = run_stats(subject_df)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats_df.to_csv(STATS_PATH, index=False)
    plot_results(channel_df, subject_df, stats_df, info, ch_names)

    print(f"included_subjects={subject_df['Pid'].nunique()}")
    print(
        "epoch_counts: "
        f"baseline_mean={trial_counts['Baseline'].mean():.2f}, "
        f"unassisted_mean={trial_counts['Unassisted'].mean():.2f}, "
        f"llm_assisted_mean={trial_counts['LLM-assisted'].mean():.2f}"
    )
    if skipped_missing_baseline:
        print(f"skipped_missing_baseline={','.join(skipped_missing_baseline)}")
    print(f"saved_stats={STATS_PATH}")
    significant_omnibus_bands = set(
        stats_df[
            (stats_df["measure"] == "raw_rPSD")
            & (stats_df["test"] == "Kruskal-Wallis")
            & (stats_df["p_fdr"] < 0.05)
        ]["band"]
    )
    significant = stats_df[
        (stats_df["test"] == "Wilcoxon signed-rank")
        & (stats_df["measure"] == "raw_rPSD")
        & (stats_df["band"].isin(significant_omnibus_bands))
        & (stats_df["p_fdr"] < 0.05)
    ]
    print("\nSignificant post-hoc Wilcoxon tests after significant Kruskal-Wallis omnibus:")
    if significant.empty:
        print("none")
    else:
        print(
            significant[
                ["measure", "band", "comparison", "statistic", "p_uncorrected", "p_fdr"]
            ].to_string(index=False)
        )

    non_normal = stats_df[
        (stats_df["measure"] == "raw_rPSD")
        & (stats_df["test"] == "Shapiro-Wilk")
        & (stats_df["p_fdr"] < 0.05)
    ]
    print("\nShapiro-Wilk normality failures, FDR-corrected:")
    if non_normal.empty:
        print("none")
    else:
        print(
            non_normal[
                ["band", "comparison", "statistic", "p_uncorrected", "p_fdr"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
