import os
import sys
import warnings
import itertools

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import mne
import numpy as np
import scipy.signal


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ERP_DIR = os.path.join(PROJECT_ROOT, "ERP")
if ERP_DIR not in sys.path:
    sys.path.insert(0, ERP_DIR)

import robust_stats as rs
from erp_shared import (
    BASE_PATH,
    BASE_PATH_ICA,
    STANDARD_CHANNELS,
    get_baseline_file_for_pid,
    get_condition_file_groups,
    load_raws_from_files,
    read_eeg_file,
)


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 50.0),
}

CONDITIONS = {
    "Control": "Unassisted",
    "Treatment": "LLM-assisted",
}

WIN_SEC = 10.0
BASELINE_FALLBACK_SEC = 60.0
COH_WIN_SEC = 0.8
TASK_START_BASELINE_PIDS = {"P3", "P11", "P12"}
EARLY_TRIGGER_MAX_SEC = 120.0
TRIGGER_BURST_GAP_SEC = 0.5
MIN_PRE_TRIGGER_BASELINE_SEC = 5.0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def available_channels(raw, requested=STANDARD_CHANNELS):
    return [ch for ch in requested if ch in raw.ch_names]


def pick_raw(raw, ch_names):
    return raw.copy().pick(ch_names)


def early_trigger_runs(raw, max_time=EARLY_TRIGGER_MAX_SEC, min_gap=TRIGGER_BURST_GAP_SEC):
    runs = []
    last_desc = None
    last_onset = None
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        onset = float(onset)
        desc = str(desc)
        if onset < 0 or onset > max_time or desc.lower() == "boundary":
            continue
        if last_desc == desc and last_onset is not None and onset - last_onset <= min_gap:
            last_onset = onset
            continue
        runs.append((desc, onset))
        last_desc = desc
        last_onset = onset
    return runs


def baseline_from_task_start(raw, pid):
    runs = early_trigger_runs(raw)
    if not runs:
        raise ValueError(f"No early non-boundary trigger found for fallback baseline in {pid}.")

    first_desc, first_onset = runs[0]
    if first_onset >= MIN_PRE_TRIGGER_BASELINE_SEC:
        tmin = 0.0
        tmax = min(first_onset, raw.times[-1])
        source = (
            f"{pid}: fallback baseline {tmin:.3f}-{tmax:.3f}s before "
            f"first trigger {first_desc}@{first_onset:.3f}s"
        )
    elif len(runs) > 1:
        second_desc, second_onset = runs[1]
        tmin = max(0.0, first_onset)
        tmax = min(second_onset, tmin + BASELINE_FALLBACK_SEC, raw.times[-1])
        source = (
            f"{pid}: fallback baseline {tmin:.3f}-{tmax:.3f}s after "
            f"beginning trigger {first_desc}@{first_onset:.3f}s before "
            f"{second_desc}@{second_onset:.3f}s"
        )
    else:
        tmin = max(0.0, first_onset)
        tmax = min(tmin + BASELINE_FALLBACK_SEC, raw.times[-1])
        source = (
            f"{pid}: fallback baseline {tmin:.3f}-{tmax:.3f}s after "
            f"only early trigger {first_desc}@{first_onset:.3f}s"
        )

    if tmax - tmin < MIN_PRE_TRIGGER_BASELINE_SEC:
        raise ValueError(
            f"Fallback baseline for {pid} is too short: {tmin:.3f}-{tmax:.3f}s."
        )
    return raw.copy().crop(tmin=tmin, tmax=tmax), source


def load_baseline_raw(pid, control_raw, ch_names):
    baseline_file = get_baseline_file_for_pid(pid, BASE_PATH_ICA)
    if baseline_file:
        raw = read_eeg_file(
            baseline_file,
            base_path=BASE_PATH_ICA,
            preload=True,
            channels=ch_names,
        )
        return raw, baseline_file
    if pid not in TASK_START_BASELINE_PIDS:
        raise FileNotFoundError(
            f"Missing baseline file for {pid} in {BASE_PATH_ICA}. "
            f"Only {sorted(TASK_START_BASELINE_PIDS)} may use task-start fallback."
        )
    raw, source = baseline_from_task_start(control_raw, pid)
    return pick_raw(raw, ch_names), source


def baseline_correct_raw(raw, baseline_raw, ch_names):
    task = pick_raw(raw, ch_names)
    base = pick_raw(baseline_raw, ch_names)
    if task.ch_names != base.ch_names:
        raise ValueError("baseline channel order mismatch")
    offsets = np.nanmean(base.get_data(), axis=1)
    task._data = task.get_data() - offsets[:, np.newaxis]
    return task


def iter_subject_raws(max_subjects=None):
    matched_pids, c_groups, t_groups = get_condition_file_groups(BASE_PATH)
    if max_subjects is not None:
        matched_pids = matched_pids[:max_subjects]

    ch_names = None
    sample_info = None
    for idx, pid in enumerate(matched_pids, 1):
        print(f"\n[{idx}/{len(matched_pids)}] {pid}: loading whole-session files", flush=True)
        raw_c = load_raws_from_files(c_groups[pid], base_path=BASE_PATH, channels=STANDARD_CHANNELS)
        raw_t = load_raws_from_files(t_groups[pid], base_path=BASE_PATH, channels=STANDARD_CHANNELS)
        if raw_c is None or raw_t is None:
            print(f"  skipping {pid}: missing condition raw")
            continue
        if ch_names is None:
            ch_names = available_channels(raw_c, STANDARD_CHANNELS)
            sample_info = pick_raw(raw_c, ch_names).info.copy()

        raw_b, baseline_source = load_baseline_raw(pid, raw_c, ch_names)
        raw_c = baseline_correct_raw(raw_c, raw_b, ch_names)
        raw_t = baseline_correct_raw(raw_t, raw_b, ch_names)
        raw_b = pick_raw(raw_b, ch_names)
        print(f"  baseline source: {baseline_source}")
        yield {
            "Pid": pid,
            "Control": raw_c,
            "Treatment": raw_t,
            "Baseline": raw_b,
            "baseline_source": baseline_source,
            "ch_names": ch_names,
            "info": sample_info,
        }


def get_subject_count(max_subjects=None):
    matched_pids, _, _ = get_condition_file_groups(BASE_PATH)
    return len(matched_pids if max_subjects is None else matched_pids[:max_subjects])


def band_masks(freqs):
    return {
        band: (freqs >= fmin) & (freqs <= fmax)
        for band, (fmin, fmax) in BANDS.items()
    }


def mean_psd_by_band(raw, ch_names):
    raw_use = pick_raw(raw, ch_names)
    spectrum = raw_use.compute_psd(method="welch", fmin=1, fmax=50, verbose=False)
    psd = spectrum.get_data()
    masks = band_masks(spectrum.freqs)
    return {
        band: psd[:, mask].mean(axis=1)
        for band, mask in masks.items()
    }, raw_use.info.copy()


def windowed_psd_by_band(raw, ch_names, baseline_band_power):
    raw_use = pick_raw(raw, ch_names)
    epochs = mne.make_fixed_length_epochs(
        raw_use,
        duration=WIN_SEC,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    if len(epochs) == 0:
        return {}
    spectrum = epochs.compute_psd(method="welch", fmin=1, fmax=50, verbose=False)
    psd = spectrum.get_data()
    masks = band_masks(spectrum.freqs)
    tiny = np.finfo(float).tiny
    out = {}
    for band, mask in masks.items():
        power = psd[:, :, mask].mean(axis=2)
        base = np.maximum(baseline_band_power[band], tiny)[None, :]
        out[band] = 10.0 * np.log10(np.maximum(power, tiny) / base)
    return out


def edge_list(ch_names):
    return list(itertools.combinations(range(len(ch_names)), 2))


def coherence_matrices(raw, ch_names):
    data = pick_raw(raw, ch_names).get_data()
    fs = float(raw.info["sfreq"])
    nperseg = max(8, int(round(COH_WIN_SEC * fs)))
    noverlap = nperseg // 2
    mats = {band: np.eye(len(ch_names)) for band in BANDS}
    for i, j in edge_list(ch_names):
        freqs, cxy = scipy.signal.coherence(
            data[i],
            data[j],
            fs=fs,
            window="hann",
            nperseg=min(nperseg, data.shape[1]),
            noverlap=min(noverlap, max(0, data.shape[1] // 2 - 1)),
        )
        for band, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            value = float(np.mean(cxy[mask])) if mask.any() else np.nan
            mats[band][i, j] = value
            mats[band][j, i] = value
    return mats


def plv_matrices(raw, ch_names):
    mats = {}
    for band, (fmin, fmax) in BANDS.items():
        raw_band = pick_raw(raw, ch_names)
        raw_band.filter(fmin, fmax, fir_design="firwin", phase="zero-double", verbose=False)
        raw_band.apply_hilbert(envelope=False, verbose=False)
        phase = np.angle(raw_band.get_data())
        mat = np.eye(len(ch_names))
        for i, j in edge_list(ch_names):
            value = np.abs(np.mean(np.exp(1j * (phase[i] - phase[j]))))
            mat[i, j] = value
            mat[j, i] = value
        mats[band] = mat
    return mats


def matrix_edge_stats(subject_mats, ch_names, measure):
    rows = []
    for band in BANDS:
        p_vals = []
        band_rows = []
        control_stack = np.asarray([s["Control"][band] for s in subject_mats])
        treatment_stack = np.asarray([s["Treatment"][band] for s in subject_mats])
        for i, j in edge_list(ch_names):
            c = control_stack[:, i, j]
            t = treatment_stack[:, i, j]
            wilc = rs.exact_wilcoxon(c, t)
            row = {
                "measure": measure,
                "band": band,
                "edge": f"{ch_names[i]}-{ch_names[j]}",
                "ch1": ch_names[i],
                "ch2": ch_names[j],
                "mean_unassisted": float(np.nanmean(c)),
                "mean_llm_assisted": float(np.nanmean(t)),
                "mean_diff": float(np.nanmean(t - c)),
                "W": wilc["stat"],
                "p": wilc["p"],
            }
            p_vals.append(wilc["p"])
            band_rows.append(row)
        q_vals = rs.fdr_bh(p_vals)
        for row, q in zip(band_rows, q_vals):
            row["q_fdr"] = q
            rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)


def sensor_positions(info, ch_names):
    picks = mne.pick_channels(info.ch_names, include=ch_names, ordered=True)
    pos = mne.channels.layout._find_topomap_coords(info, picks=picks)
    return {ch: pos[i] for i, ch in enumerate(ch_names)}


def finite_vlim(*arrays, symmetric=False):
    vals = np.concatenate([
        np.asarray(arr, float).ravel()
        for arr in arrays
        if np.asarray(arr, float).size
    ])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    if symmetric:
        vmax = float(np.nanmax(np.abs(vals)))
        vmax = vmax if vmax > 0 else 1.0
        return (-vmax, vmax)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if vmin == vmax:
        pad = abs(vmin) * 0.05 if vmin != 0 else 1.0
        return (vmin - pad, vmax + pad)
    return (vmin, vmax)


def sem(values, axis=0):
    values = np.asarray(values, float)
    n = np.sum(np.isfinite(values), axis=axis)
    sd = np.nanstd(values, axis=axis, ddof=1)
    return sd / np.sqrt(np.maximum(n, 1))


def ecdf(values):
    x = np.sort(np.asarray(values, float)[np.isfinite(values)])
    if x.size == 0:
        return x, x
    return x, np.arange(1, x.size + 1) / x.size
