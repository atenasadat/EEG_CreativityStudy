"""Main-paper result: post-insight TFR alpha/theta differences.

This script computes event-locked Morlet power around confirmed aha moments,
tests ROI-band windows, and saves the TFR ROI grid and power-profile figures.
"""

import argparse
import os
import warnings

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

warnings.filterwarnings("ignore", message="Unknown types found*")
warnings.filterwarnings("ignore", message="The data contains 'boundary' events*")
warnings.filterwarnings("ignore", message="Limited .* annotation.*")
warnings.filterwarnings("ignore", message="Concatenation of Annotations.*")
warnings.filterwarnings("ignore", message="Data file name in EEG.data .*")

import mne
import numpy as np
import scipy.stats as stats
from mne.stats import combine_adjacency, permutation_cluster_1samp_test

from erp_shared import (
    BASE_PATH,
    get_trigger_file_groups,
    load_aha_epochs_from_files,
)


mne.set_log_level("ERROR")

ROIS = {
    "Cluster51": ["P3", "C3", "F3"],
    "Frontal": ["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8"],
    "Central": ["C3", "Cz", "C4"],
    "Parietal": ["P3", "Pz", "P4", "P7", "P8"],
    "Occipital": ["O1", "O2"],
    "Temporal": ["T7", "T8"],
}

CONTROL_LABEL = "Unassisted"
TREATMENT_LABEL = "LLM-assisted"
TFR_GRID_ROIS = ["Frontal", "Central", "Parietal", "Occipital", "Temporal"]
CHANNELS = sorted({ch for channels in ROIS.values() for ch in channels})
FREQS = np.arange(1.0, 50.0, 2.0)
CYCL_DIV = 2.0
EPOCH_WINDOW = (-2.5, 5)
BASELINE_WINDOW = (-2.5, -2.0)
SEARCH_WINDOW = (0.0, 4.0)
DECIM = 2
N_PERMUTATIONS = 500
ALPHA = 0.05
SEED = 0

ROI_BAND_WINDOWS = [
    ("Parietal", "Theta", "Post_0_4", 4, 8, 0.0, 4.0),
    ("Parietal", "Theta", "Post_1_2", 4, 8, 1.0, 2.0),
    ("Parietal", "Theta", "Post_2_3", 4, 8, 2.0, 3.0),
    ("Frontal", "Alpha", "Post_0_1", 8, 13, 0.0, 1.0),
]

POWER_PLOT_COMPARISONS = [
    ("Parietal", "Theta", "Post_0_4", 4, 8, 0.0, 4.0),
    ("Frontal", "Alpha", "Post_0_1", 8, 13, 0.0, 1.0),
]
DEFAULT_PLOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "tfr_power_profiles.png",
)
DEFAULT_TFR_GRID_PATH = os.path.join(
    os.path.dirname(__file__),
    "tfr_roi_condition_grid.png",
)


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


def compute_subject_tfr_db(epochs):
    ep = epochs.copy().pick(CHANNELS)
    tfr = ep.compute_tfr(
        method="morlet",
        freqs=FREQS,
        n_cycles=FREQS / CYCL_DIV,
        return_itc=False,
        average=False,
        decim=DECIM,
        verbose=False,
    )

    baseline_mask = (tfr.times >= BASELINE_WINDOW[0]) & (tfr.times <= BASELINE_WINDOW[1])
    search_mask = (tfr.times >= SEARCH_WINDOW[0]) & (tfr.times <= SEARCH_WINDOW[1])
    if not baseline_mask.any() or not search_mask.any():
        raise ValueError("Baseline/search window has no TFR samples.")

    power = np.maximum(tfr.data, np.finfo(float).tiny)
    baseline = power[:, :, :, baseline_mask].mean(axis=3, keepdims=True)
    db_power = 10.0 * np.log10(power / np.maximum(baseline, np.finfo(float).tiny))

    subject_tfr = db_power[:, :, :, search_mask].mean(axis=0)
    return subject_tfr, tfr.times[search_mask], ep.ch_names


def roi_average(subject_tfr, ch_names, roi_channels):
    picks = [ch_names.index(ch) for ch in roi_channels if ch in ch_names]
    if not picks:
        return None
    return subject_tfr[picks].mean(axis=0)


def build_subject_differences(roi_names):
    matched_pids, c_groups, t_groups = get_trigger_file_groups(BASE_PATH, verbose=False)
    control_by_roi = {roi_name: [] for roi_name in roi_names}
    treatment_by_roi = {roi_name: [] for roi_name in roi_names}
    included = []
    trial_counts = []
    times = None
    print(f"matched_subjects={len(matched_pids)}")

    for idx, pid in enumerate(matched_pids, start=1):
        print(f"processing_subject={idx}/{len(matched_pids)} pid={pid}", flush=True)
        ep_c = load_aha_epochs_from_files(
            c_groups[pid],
            base_path=BASE_PATH,
            tmin=EPOCH_WINDOW[0],
            tmax=EPOCH_WINDOW[1],
            baseline=None,
        )
        ep_t = load_aha_epochs_from_files(
            t_groups[pid],
            base_path=BASE_PATH,
            tmin=EPOCH_WINDOW[0],
            tmax=EPOCH_WINDOW[1],
            baseline=None,
        )
        if ep_c is None or ep_t is None:
            continue

        control_all, these_times, ch_names = compute_subject_tfr_db(ep_c)
        treatment_all, _, treatment_ch_names = compute_subject_tfr_db(ep_t)
        if ch_names != treatment_ch_names:
            continue
        if control_all.shape != treatment_all.shape:
            continue

        roi_values = {}
        for roi_name in roi_names:
            control = roi_average(control_all, ch_names, ROIS[roi_name])
            treatment = roi_average(treatment_all, ch_names, ROIS[roi_name])
            if control is None or treatment is None:
                break
            roi_values[roi_name] = (control, treatment)
        else:
            if times is None:
                times = these_times
            elif len(times) != len(these_times) or not np.allclose(times, these_times):
                continue

            for roi_name, (control, treatment) in roi_values.items():
                control_by_roi[roi_name].append(control)
                treatment_by_roi[roi_name].append(treatment)
            included.append(pid)
            trial_counts.append((pid, len(ep_c), len(ep_t)))
            print(f"included_subjects={len(included)}", flush=True)
            continue

        continue

    if not included:
        raise RuntimeError("No matched subjects found.")

    return (
        {roi_name: np.asarray(rows) for roi_name, rows in control_by_roi.items()},
        {roi_name: np.asarray(rows) for roi_name, rows in treatment_by_roi.items()},
        included,
        trial_counts,
        times,
    )


def cluster_rows(t_obs, clusters, p_values, times, alpha):
    rows = []
    for cluster_id, p_val in enumerate(p_values):
        mask = clusters[cluster_id]
        freq_inds, time_inds = np.where(mask)
        if not len(freq_inds):
            continue

        cluster_t = t_obs[mask]
        peak_idx = np.argmax(np.abs(cluster_t))
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "p_value": float(p_val),
                "significant": bool(p_val < alpha),
                "freq_start": float(FREQS[freq_inds.min()]),
                "freq_end": float(FREQS[freq_inds.max()]),
                "time_start": float(times[time_inds.min()]),
                "time_end": float(times[time_inds.max()]),
                "mean_t": float(cluster_t.mean()),
                "peak_freq": float(FREQS[freq_inds[peak_idx]]),
                "peak_time": float(times[time_inds[peak_idx]]),
                "direction": f"{TREATMENT_LABEL} > {CONTROL_LABEL}"
                if cluster_t.mean() > 0
                else f"{CONTROL_LABEL} > {TREATMENT_LABEL}",
            }
        )
    return sorted(rows, key=lambda row: row["p_value"])


def run_fdr_wilcoxon(control, treatment, times, alpha):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wilcoxon = stats.wilcoxon(
            treatment,
            control,
            axis=0,
            zero_method="wilcox",
            method="auto",
            nan_policy="omit",
        )
    w_obs = wilcoxon.statistic
    p_unc = wilcoxon.pvalue
    p_fdr = fdr_bh(p_unc.ravel()).reshape(p_unc.shape)
    mean_diff = np.nanmean(treatment - control, axis=0)
    sig = np.isfinite(p_fdr) & (p_fdr < alpha)

    print("test=pointwise Wilcoxon signed-rank with FDR")
    print(f"significant_pixels={int(sig.sum())}")

    flat_order = np.lexsort((
        np.nan_to_num(p_unc.ravel(), nan=np.inf),
        np.nan_to_num(p_fdr.ravel(), nan=np.inf),
    ))
    shown = 0
    for flat_idx in flat_order:
        f_idx, t_idx = np.unravel_index(flat_idx, p_fdr.shape)
        if not np.isfinite(p_fdr[f_idx, t_idx]):
            continue
        direction = (
            f"{TREATMENT_LABEL} > {CONTROL_LABEL}"
            if mean_diff[f_idx, t_idx] > 0
            else f"{CONTROL_LABEL} > {TREATMENT_LABEL}"
        )
        print(
            f"pixel: f={FREQS[f_idx]:.0f}Hz time={times[t_idx]:.3f}s "
            f"W={w_obs[f_idx, t_idx]:.3f} p={p_unc[f_idx, t_idx]:.4g} "
            f"fdr={p_fdr[f_idx, t_idx]:.4g} {direction}"
        )
        shown += 1
        if shown >= 10:
            break


def band_window_roi_names():
    return sorted({roi for roi, *_ in ROI_BAND_WINDOWS})


def mean_sem(data):
    mean = np.nanmean(data, axis=0)
    n = np.sum(np.isfinite(data), axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, sem


def smooth_line(values, window=5):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_power_profiles(control_by_roi, treatment_by_roi, times, output_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = os.path.abspath(output_path)
    plt.rcParams.update({"font.family": "Times New Roman"})
    colors = {CONTROL_LABEL: "#0057B8", TREATMENT_LABEL: "#D00000"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), constrained_layout=True)

    for row_idx, (roi_name, band, window, fmin, fmax, tmin, tmax) in enumerate(
        POWER_PLOT_COMPARISONS
    ):
        control = control_by_roi[roi_name]
        treatment = treatment_by_roi[roi_name]
        freq_mask = (FREQS >= fmin) & (FREQS <= fmax)
        time_mask = (times >= tmin) & (times <= tmax)

        control_by_freq = control[:, :, time_mask].mean(axis=2)
        treatment_by_freq = treatment[:, :, time_mask].mean(axis=2)
        control_by_time = control[:, freq_mask, :].mean(axis=1)
        treatment_by_time = treatment[:, freq_mask, :].mean(axis=1)

        c_freq_mean, c_freq_sem = mean_sem(control_by_freq)
        t_freq_mean, t_freq_sem = mean_sem(treatment_by_freq)
        c_time_mean, c_time_sem = mean_sem(control_by_time)
        t_time_mean, t_time_sem = mean_sem(treatment_by_time)
        c_freq_mean, c_freq_sem = smooth_line(c_freq_mean), smooth_line(c_freq_sem)
        t_freq_mean, t_freq_sem = smooth_line(t_freq_mean), smooth_line(t_freq_sem)
        c_time_mean, c_time_sem = smooth_line(c_time_mean), smooth_line(c_time_sem)
        t_time_mean, t_time_sem = smooth_line(t_time_mean), smooth_line(t_time_sem)

        freq_ax = axes[row_idx, 0]
        time_ax = axes[row_idx, 1]

        for edge in (fmin, fmax):
            freq_ax.axvline(edge, color="0.35", linestyle="--", linewidth=1.2, alpha=0.45)
        freq_ax.plot(
            FREQS,
            c_freq_mean,
            color=colors[CONTROL_LABEL],
            label=CONTROL_LABEL,
            linewidth=2.2,
        )
        freq_ax.fill_between(
            FREQS,
            c_freq_mean - c_freq_sem,
            c_freq_mean + c_freq_sem,
            color=colors[CONTROL_LABEL],
            alpha=0.12,
        )
        freq_ax.plot(
            FREQS,
            t_freq_mean,
            color=colors[TREATMENT_LABEL],
            label=TREATMENT_LABEL,
            linewidth=2.2,
        )
        freq_ax.fill_between(
            FREQS,
            t_freq_mean - t_freq_sem,
            t_freq_mean + t_freq_sem,
            color=colors[TREATMENT_LABEL],
            alpha=0.12,
        )
        freq_ax.set_xlabel("Frequency (Hz)")
        freq_ax.set_ylabel("Power (db)")
        freq_ax.set_xlim(left=0)
        freq_ax.legend(frameon=False, loc="lower right")

        for edge in (tmin, tmax):
            time_ax.axvline(edge, color="0.35", linestyle="--", linewidth=1.2, alpha=0.45)
        time_ax.plot(
            times,
            c_time_mean,
            color=colors[CONTROL_LABEL],
            label=CONTROL_LABEL,
            linewidth=2.2,
        )
        time_ax.fill_between(
            times,
            c_time_mean - c_time_sem,
            c_time_mean + c_time_sem,
            color=colors[CONTROL_LABEL],
            alpha=0.12,
        )
        time_ax.plot(
            times,
            t_time_mean,
            color=colors[TREATMENT_LABEL],
            label=TREATMENT_LABEL,
            linewidth=2.2,
        )
        time_ax.fill_between(
            times,
            t_time_mean - t_time_sem,
            t_time_mean + t_time_sem,
            color=colors[TREATMENT_LABEL],
            alpha=0.12,
        )
        time_ax.set_xlabel("Time (s)")
        time_ax.set_ylabel("Power (db)")
        time_ax.set_xlim(left=0, right=times[-1])
        time_ax.legend(frameon=False, loc="lower right")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_plot={output_path}")


def plot_tfr_roi_grid(control_by_roi, treatment_by_roi, times, output_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = os.path.abspath(output_path)
    plt.rcParams.update({"font.family": "Times New Roman"})

    plotted = []
    for roi_name in TFR_GRID_ROIS:
        if roi_name not in control_by_roi:
            continue
        plotted.append((roi_name, np.nanmean(control_by_roi[roi_name], axis=0), np.nanmean(treatment_by_roi[roi_name], axis=0)))

    if not plotted:
        print("no_tfr_grid_rois")
        return

    all_values = np.concatenate([arr.ravel() for _, control, treatment in plotted for arr in (control, treatment)])
    vmax = np.nanpercentile(np.abs(all_values), 98)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(2, len(plotted), figsize=(15, 6), constrained_layout=True)
    if len(plotted) == 1:
        axes = axes.reshape(2, 1)

    extent = [times[0], times[-1], FREQS[0], FREQS[-1]]
    images = []
    for col, (roi_name, control, treatment) in enumerate(plotted):
        for row, data in enumerate((control, treatment)):
            ax = axes[row, col]
            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            images.append(im)
            ax.set_xlim(left=0, right=times[-1])
            ax.set_xlabel("Time from event (s)")
            if col == 0:
                ax.set_ylabel(f"{CONTROL_LABEL}\nFrequency (Hz)" if row == 0 else f"{TREATMENT_LABEL}\nFrequency (Hz)")
            if row == 0:
                ax.set_title(roi_name)

    cbar = fig.colorbar(images[0], ax=axes, shrink=0.9, location="right")
    cbar.set_label("Power (db)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_tfr_grid={output_path}")


def print_trial_summary(trial_counts, times):
    counts = np.asarray([(control, treatment) for _, control, treatment in trial_counts])
    print(
        "trial_counts: "
        f"{CONTROL_LABEL} mean={counts[:, 0].mean():.2f}, range={counts[:, 0].min()}-{counts[:, 0].max()}; "
        f"{TREATMENT_LABEL} mean={counts[:, 1].mean():.2f}, range={counts[:, 1].min()}-{counts[:, 1].max()}"
    )
    print(f"tfr_samples: subjects={len(trial_counts)}, freqs={len(FREQS)}, times={len(times)}")


def run_band_window_wilcoxon(control_by_roi, treatment_by_roi, times, included, trial_counts, alpha):
    rows = []
    print("test=ROI band/window Wilcoxon signed-rank with FDR")
    print(f"subjects={len(included)}")
    print_trial_summary(trial_counts, times)

    for roi_name in band_window_roi_names():
        if roi_name not in control_by_roi:
            continue
        print(
            f"roi={roi_name} channels={','.join(ROIS[roi_name])} "
            f"input_shape={control_by_roi[roi_name].shape}"
        )

    for roi_name, band, window, fmin, fmax, tmin, tmax in ROI_BAND_WINDOWS:
        if roi_name not in control_by_roi:
            continue

        control = control_by_roi[roi_name]
        treatment = treatment_by_roi[roi_name]
        freq_mask = (FREQS >= fmin) & (FREQS <= fmax)
        if not freq_mask.any():
            continue

        time_mask = (times >= tmin) & (times <= tmax)
        if not time_mask.any():
            continue

        c = control[:, freq_mask, :][:, :, time_mask].mean(axis=(1, 2))
        t = treatment[:, freq_mask, :][:, :, time_mask].mean(axis=(1, 2))
        diff = t - c
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.wilcoxon(
                t,
                c,
                zero_method="wilcox",
                method="auto",
                nan_policy="omit",
            )
        rows.append({
            "roi": roi_name,
            "band": band,
            "window": window,
            "freq_start": fmin,
            "freq_end": fmax,
            "time_start": tmin,
            "time_end": tmax,
            "mean_diff": float(np.nanmean(diff)),
            "wilcoxon_w": float(res.statistic),
            "p_unc": float(res.pvalue),
            "direction": f"{TREATMENT_LABEL} > {CONTROL_LABEL}"
            if np.nanmean(diff) > 0
            else f"{CONTROL_LABEL} > {TREATMENT_LABEL}",
        })

    if not rows:
        print("no_valid_band_windows")
        return

    p_fdr = fdr_bh([row["p_unc"] for row in rows])
    for row, p_val in zip(rows, p_fdr):
        row["p_fdr"] = float(p_val)

    significant = [row for row in rows if row["p_fdr"] < alpha]

    print(f"tests={len(rows)}")
    print(f"significant_tests={len(significant)}")
    for row in rows:
        print(
            f"{row['roi']} {row['band']} {row['window']}: "
            f"{row['freq_start']:.0f}-{row['freq_end']:.0f}Hz, "
            f"{row['time_start']:.1f}-{row['time_end']:.1f}s, "
            f"W={row['wilcoxon_w']:.1f}, p={row['p_unc']:.4g}, "
            f"fdr={row['p_fdr']:.4g}, {row['direction']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi", choices=sorted(ROIS), action="append")
    parser.add_argument(
        "--test",
        choices=["cluster", "fdr", "wilcoxon", "band-wilcoxon", "both"],
        default="band-wilcoxon",
    )
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--cluster-alpha", type=float, default=ALPHA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", default=DEFAULT_PLOT_PATH)
    parser.add_argument("--tfr-grid-path", default=DEFAULT_TFR_GRID_PATH)
    args = parser.parse_args()

    if args.roi:
        roi_names = args.roi
    elif args.test == "band-wilcoxon" and args.plot:
        roi_names = sorted(set(band_window_roi_names()) | set(TFR_GRID_ROIS))
    elif args.test == "band-wilcoxon":
        roi_names = band_window_roi_names()
    else:
        roi_names = list(ROIS)

    print("TFR ROI test")
    print(f"baseline={BASELINE_WINDOW[0]} to {BASELINE_WINDOW[1]} s")
    print(f"search={SEARCH_WINDOW[0]} to {SEARCH_WINDOW[1]} s")

    control_by_roi, treatment_by_roi, included, trial_counts, times = build_subject_differences(roi_names)

    if args.test == "band-wilcoxon":
        run_band_window_wilcoxon(
            control_by_roi,
            treatment_by_roi,
            times,
            included,
            trial_counts,
            args.alpha,
        )
        if args.plot:
            plot_power_profiles(control_by_roi, treatment_by_roi, times, args.plot_path)
            plot_tfr_roi_grid(control_by_roi, treatment_by_roi, times, args.tfr_grid_path)
        return

    if args.test == "both":
        run_band_window_wilcoxon(
            control_by_roi,
            treatment_by_roi,
            times,
            included,
            trial_counts,
            args.alpha,
        )
        if args.plot:
            plot_power_profiles(control_by_roi, treatment_by_roi, times, args.plot_path)
            plot_tfr_roi_grid(control_by_roi, treatment_by_roi, times, args.tfr_grid_path)

    for roi_name in roi_names:
        roi_channels = ROIS[roi_name]
        print(f"\nroi={roi_name} channels={','.join(roi_channels)}", flush=True)
        control = control_by_roi[roi_name]
        treatment = treatment_by_roi[roi_name]
        X = treatment - control
        print(f"subjects={len(included)}")
        print(f"input_shape={X.shape}  # subjects x freqs x times")

        if args.test in ("fdr", "wilcoxon", "both"):
            run_fdr_wilcoxon(control, treatment, times, args.alpha)

        if args.test in ("cluster", "both"):
            threshold = stats.t.ppf(1 - args.cluster_alpha / 2, df=X.shape[0] - 1)
            adjacency = combine_adjacency(len(FREQS), len(times))

            print("starting_permutation", flush=True)
            t_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
                X,
                threshold=threshold,
                tail=0,
                n_permutations=args.n_permutations,
                adjacency=adjacency,
                out_type="mask",
                seed=args.seed,
                verbose=False,
            )
            print("finished_permutation", flush=True)
            rows = cluster_rows(t_obs, clusters, p_values, times, args.alpha)
            significant = [row for row in rows if row["significant"]]

            print(f"significant_clusters={len(significant)}")

            for row in significant:
                print(
                    f"cluster {row['cluster_id']}: p={row['p_value']:.4f}, "
                    f"{row['freq_start']:.0f}-{row['freq_end']:.0f} Hz, "
                    f"{row['time_start']:.3f}-{row['time_end']:.3f} s, "
                    f"{row['direction']}"
                )


if __name__ == "__main__":
    main()
