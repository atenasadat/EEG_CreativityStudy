"""Main-paper result support: directed connectivity with dDTF.

This script computes whole-session ROI-level directed transfer measures and
provides ROI definitions/functions used by the Wallas dDTF analysis.
"""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal

from wholesession_shared import (
    BASE_PATH,
    STANDARD_CHANNELS,
    available_channels,
    baseline_correct_raw,
    ensure_dir,
    get_condition_file_groups,
    load_baseline_raw,
    read_eeg_file,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ERP_DIR = os.path.join(PROJECT_ROOT, "ERP")
if ERP_DIR not in sys.path:
    sys.path.insert(0, ERP_DIR)
import robust_stats as rs


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "wholesession_ddtf_stats")

ROIS = {
    "Frontal": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "Parietal": ["P7", "P3", "Pz", "P4", "P8"],
    "Occipital": ["O1", "O2"],
}

DDTF_BANDS = {
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
}


def roi_series(raw, ch_names):
    data = raw.copy().pick(ch_names).get_data()
    rows = []
    names = []
    for roi, chans in ROIS.items():
        idx = [ch_names.index(ch) for ch in chans if ch in ch_names]
        if not idx:
            continue
        rows.append(np.nanmean(data[idx], axis=0))
        names.append(roi)
    return np.asarray(rows), names


def boundary_samples(raw):
    sfreq = float(raw.info["sfreq"])
    out = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        if "boundary" in str(desc).lower():
            sample = int(round(float(onset) * sfreq))
            if 0 < sample < raw.n_times:
                out.append(sample)
    return sorted(set(out))


def window_starts(raw, window_sec):
    sfreq = float(raw.info["sfreq"])
    n = int(round(window_sec * sfreq))
    starts = []
    edges = [0] + boundary_samples(raw) + [raw.n_times]
    for a, b in zip(edges[:-1], edges[1:]):
        starts.extend(range(a, b - n + 1, n))
    return starts, n


def fit_mvar(x, order):
    n_vars, n_times = x.shape
    if n_times <= order + n_vars * order + 1:
        return None, None, False

    y = x[:, order:].T
    z = np.column_stack([x[:, order - lag : n_times - lag].T for lag in range(1, order + 1)])
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)

    a = np.zeros((order, n_vars, n_vars), dtype=float)
    for lag in range(order):
        a[lag] = coef[lag * n_vars : (lag + 1) * n_vars, :].T

    resid = y - z @ coef
    sigma = np.cov(resid, rowvar=False)
    if sigma.ndim == 0:
        sigma = np.eye(n_vars) * float(sigma)
    reg = max(np.trace(sigma) / n_vars, np.finfo(float).eps) * 1e-8
    sigma = sigma + np.eye(n_vars) * reg

    comp = np.zeros((n_vars * order, n_vars * order), dtype=float)
    comp[:n_vars, :] = np.concatenate(a, axis=1)
    if order > 1:
        comp[n_vars:, :-n_vars] = np.eye(n_vars * (order - 1))
    stable = bool(np.max(np.abs(np.linalg.eigvals(comp))) < 1.0)
    return a, sigma, stable


def ddtf_window(x, fs, order, freqs, band_masks):
    x = signal.detrend(x, axis=1, type="linear")
    x = x - x.mean(axis=1, keepdims=True)
    scale = x.std(axis=1, keepdims=True)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        return None, False
    x = x / scale

    a, sigma, stable = fit_mvar(x, order)
    if a is None:
        return None, False

    n_vars = x.shape[0]
    eye = np.eye(n_vars, dtype=complex)
    spec = np.zeros((len(freqs), n_vars, n_vars), dtype=float)
    for fi, freq in enumerate(freqs):
        af = eye.copy()
        for lag, amat in enumerate(a, start=1):
            af -= amat * np.exp(-2j * np.pi * freq * lag / fs)
        try:
            h = np.linalg.inv(af)
        except np.linalg.LinAlgError:
            h = np.linalg.pinv(af)

        denom = np.sqrt(np.sum(np.abs(h) ** 2, axis=1, keepdims=True))
        dtf = np.divide(np.abs(h), denom, out=np.zeros_like(np.abs(h)), where=denom > 0)

        s = h @ sigma @ h.conj().T
        try:
            p = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            p = np.linalg.pinv(s)
        pc_denom = np.sqrt(np.abs(np.diag(p))[:, None] * np.abs(np.diag(p))[None, :])
        pc = np.divide(np.abs(p), pc_denom, out=np.zeros_like(np.abs(p)), where=pc_denom > 0)
        spec[fi] = dtf * pc

    out = {}
    for band, mask in band_masks.items():
        out[band] = np.nanmean(spec[mask], axis=0)
    return out, stable


def ddtf_raw(raw, ch_names, window_sec, resample_sfreq, order):
    data, roi_names = roi_series(raw, ch_names)
    starts, n = window_starts(raw, window_sec)
    orig_sfreq = float(raw.info["sfreq"])
    out = {band: [] for band in DDTF_BANDS}
    used_count = 0
    unstable_count = 0
    failed_count = 0

    freqs = np.arange(1.0, 50.0 + 0.5, 0.5)
    band_masks = {
        band: (freqs >= lo) & (freqs < hi)
        for band, (lo, hi) in DDTF_BANDS.items()
    }

    for start in starts:
        win = data[:, start : start + n]
        if win.shape[1] != n:
            continue
        x = signal.resample_poly(win, int(resample_sfreq), int(round(orig_sfreq)), axis=1)
        vals, stable = ddtf_window(x, resample_sfreq, order, freqs, band_masks)
        if vals is None:
            failed_count += 1
            continue
        if not stable:
            unstable_count += 1
            continue
        used_count += 1
        for band in DDTF_BANDS:
            out[band].append(vals[band])

    mats = {}
    for band, values in out.items():
        mats[band] = np.nanmean(np.asarray(values), axis=0) if values else np.full(
            (len(roi_names), len(roi_names)), np.nan
        )
    coverage = {
        "total_sec": raw.n_times / orig_sfreq,
        "included_sec": used_count * window_sec,
        "excluded_sec": raw.n_times / orig_sfreq - used_count * window_sec,
        "included_pct": 100 * used_count * window_sec / (raw.n_times / orig_sfreq),
        "n_candidate_windows": len(starts),
        "n_windows": used_count,
        "n_unstable_windows": unstable_count,
        "n_failed_windows": failed_count,
        "stable_pct": 100 * used_count / len(starts) if starts else np.nan,
    }
    return mats, roi_names, coverage


def empty_subject_acc(roi_names):
    return {
        band: np.zeros((len(roi_names), len(roi_names)), dtype=float)
        for band in DDTF_BANDS
    }, {band: 0.0 for band in DDTF_BANDS}


def process_condition(files, pid, baseline_raw, ch_names, window_sec, resample_sfreq, order):
    roi_names = list(ROIS)
    acc, weights = empty_subject_acc(roi_names)
    coverage_rows = []
    for fname in files:
        raw = read_eeg_file(fname, base_path=BASE_PATH, preload=True, channels=ch_names)
        raw = baseline_correct_raw(raw, baseline_raw, ch_names)
        mats, roi_names, cov = ddtf_raw(raw, ch_names, window_sec, resample_sfreq, order)
        cov.update({"Pid": pid, "file": fname})
        coverage_rows.append(cov)
        for band in DDTF_BANDS:
            if np.isfinite(mats[band]).any() and cov["n_windows"] > 0:
                acc[band] += mats[band] * cov["n_windows"]
                weights[band] += cov["n_windows"]

    subject_mats = {
        band: acc[band] / weights[band] if weights[band] else np.full(acc[band].shape, np.nan)
        for band in DDTF_BANDS
    }
    return subject_mats, roi_names, coverage_rows


def subject_feature_rows(subjects, roi_names):
    rows = []
    for subj in subjects:
        for cond in ["Control", "Treatment"]:
            row = {"Pid": subj["Pid"], "condition": cond, "baseline_source": subj["baseline_source"]}
            for band in DDTF_BANDS:
                mat = subj[cond][band]
                for src_i, source in enumerate(roi_names):
                    for tgt_i, target in enumerate(roi_names):
                        if src_i == tgt_i:
                            continue
                        row[f"{band} {source}->{target} dDTF"] = float(mat[tgt_i, src_i])
            rows.append(row)
    return pd.DataFrame(rows)


def stats_table(subjects, roi_names):
    rows = []
    for band in DDTF_BANDS:
        p_vals = []
        band_rows = []
        for src_i, source in enumerate(roi_names):
            for tgt_i, target in enumerate(roi_names):
                if src_i == tgt_i:
                    continue
                c = np.asarray([s["Control"][band][tgt_i, src_i] for s in subjects], dtype=float)
                t = np.asarray([s["Treatment"][band][tgt_i, src_i] for s in subjects], dtype=float)
                wilc = rs.exact_wilcoxon(c, t)
                mask = np.isfinite(c) & np.isfinite(t)
                band_rows.append({
                    "band": band,
                    "source": source,
                    "target": target,
                    "direction": f"{source}->{target}",
                    "n": int(mask.sum()),
                    "mean_unassisted": float(np.nanmean(c)),
                    "mean_llm_assisted": float(np.nanmean(t)),
                    "mean_diff": float(np.nanmean(t - c)),
                    "W": wilc["stat"],
                    "p": wilc["p"],
                    "method": wilc["method"],
                })
                p_vals.append(wilc["p"])
        q_vals = rs.fdr_bh(p_vals)
        for row, q in zip(band_rows, q_vals):
            row["q_fdr_within_band"] = q
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_fdr_all"] = rs.fdr_bh(out["p"])
    return out


def plot_heatmaps(subjects, roi_names):
    means = {}
    for cond in ["Control", "Treatment"]:
        means[cond] = {}
        for band in DDTF_BANDS:
            means[cond][band] = np.nanmean(np.asarray([s[cond][band] for s in subjects]), axis=0)

    fig, axes = plt.subplots(len(DDTF_BANDS), 3, figsize=(10, 6), dpi=250, constrained_layout=True)
    if len(DDTF_BANDS) == 1:
        axes = axes[None, :]
    for r, band in enumerate(DDTF_BANDS):
        c = means["Control"][band].copy()
        t = means["Treatment"][band].copy()
        diff = t - c
        for mat in [c, t, diff]:
            np.fill_diagonal(mat, np.nan)
        vmax = np.nanmax([np.nanmax(c), np.nanmax(t)])
        dlim = np.nanmax(np.abs(diff))
        panels = [
            (c, "Unassisted", "magma", 0, vmax),
            (t, "LLM-assisted", "magma", 0, vmax),
            (diff, "LLM - Unassisted", "RdBu_r", -dlim, dlim),
        ]
        for ax, (mat, title, cmap, vmin, vmax_) in zip(axes[r], panels):
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax_)
            ax.set_title(f"{band} {title}", fontsize=9)
            ax.set_xticks(range(len(roi_names)))
            ax.set_yticks(range(len(roi_names)))
            ax.set_xticklabels(roi_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(roi_names, fontsize=8)
            ax.set_xlabel("Source")
            ax.set_ylabel("Target")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(os.path.join(OUTPUT_DIR, "whole_session_ddtf_heatmaps.png"), bbox_inches="tight")
    plt.close(fig)


def run(max_subjects=None, window_sec=4.0, resample_sfreq=128.0, order=10):
    ensure_dir(OUTPUT_DIR)
    matched_pids, c_groups, t_groups = get_condition_file_groups(BASE_PATH)
    if max_subjects is not None:
        matched_pids = matched_pids[:max_subjects]

    print("\nWHOLE-SESSION dDTF")
    print(f"Matched subjects: N={len(matched_pids)}")
    print("Time included: all full clean 4-sec windows, including uncoded Wallas time.")

    subjects = []
    coverage = []
    roi_names = list(ROIS)
    for idx, pid in enumerate(matched_pids, 1):
        print(f"\n[{idx}/{len(matched_pids)}] {pid}", flush=True)
        first_raw = read_eeg_file(c_groups[pid][0], base_path=BASE_PATH, preload=True, channels=STANDARD_CHANNELS)
        ch_names = available_channels(first_raw, STANDARD_CHANNELS)
        baseline_raw, baseline_source = load_baseline_raw(pid, first_raw, ch_names)
        print(f"  baseline source: {baseline_source}")

        control_mats, roi_names, cov_c = process_condition(
            c_groups[pid], pid, baseline_raw, ch_names, window_sec, resample_sfreq, order
        )
        treatment_mats, roi_names, cov_t = process_condition(
            t_groups[pid], pid, baseline_raw, ch_names, window_sec, resample_sfreq, order
        )
        for row in cov_c:
            row["condition"] = "Control"
        for row in cov_t:
            row["condition"] = "Treatment"
        coverage.extend(cov_c + cov_t)
        subjects.append({
            "Pid": pid,
            "baseline_source": baseline_source,
            "Control": control_mats,
            "Treatment": treatment_mats,
        })
        print(
            "  windows: control=%d treatment=%d"
            % (
                sum(r["n_windows"] for r in cov_c),
                sum(r["n_windows"] for r in cov_t),
            )
        )

    subject_df = subject_feature_rows(subjects, roi_names)
    stats_df = stats_table(subjects, roi_names)
    coverage_df = pd.DataFrame(coverage)

    subject_df.to_csv(os.path.join(OUTPUT_DIR, "subject_whole_session_ddtf.csv"), index=False)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "whole_session_ddtf_stats.csv"), index=False)
    coverage_df.to_csv(os.path.join(OUTPUT_DIR, "whole_session_ddtf_coverage.csv"), index=False)
    plot_heatmaps(subjects, roi_names)

    total_sec = coverage_df["total_sec"].sum()
    included_sec = coverage_df["included_sec"].sum()
    print("\nCoverage:")
    print(f"  included: {included_sec:.1f} / {total_sec:.1f} sec ({100 * included_sec / total_sec:.2f}%)")
    print(f"  excluded leftover/boundary: {total_sec - included_sec:.1f} sec ({100 * (total_sec - included_sec) / total_sec:.2f}%)")

    print("\nSmallest p-values:")
    cols = ["band", "direction", "mean_unassisted", "mean_llm_assisted", "mean_diff", "p", "q_fdr_within_band", "q_fdr_all"]
    print(stats_df.sort_values("p")[cols].head(12).to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nDONE. Outputs saved to: {OUTPUT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description="Whole-session ROI dDTF analysis.")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--resample-sfreq", type=float, default=128.0)
    parser.add_argument("--order", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        max_subjects=args.max_subjects,
        window_sec=args.window_sec,
        resample_sfreq=args.resample_sfreq,
        order=args.order,
    )
