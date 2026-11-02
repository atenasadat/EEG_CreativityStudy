"""PCA region signals for the dDTF in wallas_analysis/wallas_mapping.ipynb (Tak's suggestion).

Replaces the channel average in `whole_session_ddtf.roi_series` with the first principal component of each
region's channels. The weights are fitted once per participant on all of that participant's task recordings
from BOTH conditions (1-45 Hz band-passed copy, for the fit only), so both conditions use the same spatial
filter. Weights have unit norm, and their sign is chosen so they sum to a positive number. dDTF does not
depend on the sign or the scale of a region signal, because each window is z-scored before the MVAR fit.

How to use it in the notebook (cell "12. Question 1 dDTF tests", step "# 1. collect every ROI window"):

    import pca_roi_series as P
    weights = P.fit_participant_weights(eeg, CLEAN_DIR, CHANNELS, whole_session_ddtf.ROIS)   # once, before the loop
    ...
    # in the loop, replace
    #     data, roi_names = whole_session_ddtf.roi_series(raw, CHANNELS)
    # with
    data, roi_names = P.roi_series_pca(raw, CHANNELS, weights[pid])

Everything else in the notebook stays the same. Check: with equal weights (P.equal_weights) the result is
identical to the channel average, which confirms the swap is wired correctly.
"""
import numpy as np
import mne


def _pc1(data, idx):
    x = data[idx] - data[idx].mean(axis=1, keepdims=True)
    evals, evecs = np.linalg.eigh(np.cov(x))
    w = evecs[:, -1]
    w = w if w.sum() >= 0 else -w
    return w / np.linalg.norm(w), float(100 * evals[-1] / evals.sum())


def fit_participant_weights(eeg, clean_dir, channels, rois):
    """eeg: the notebook's dict {(pid, cond, task): {'file': ...}}. Returns {pid: {roi: (idx, w, pc1_var_pct)}}."""
    by_pid = {}
    for (pid, cond, task), e in eeg.items():
        by_pid.setdefault(pid, []).append(e['file'])
    out = {}
    for pid, files in by_pid.items():
        chunks = []
        for f in files:
            raw = mne.io.read_raw_eeglab(str(clean_dir / f), preload=True, verbose=False).pick(channels)
            raw.filter(1.0, 45.0, verbose='ERROR')
            chunks.append(raw.get_data())
        data = np.concatenate(chunks, axis=1)
        out[pid] = {}
        for roi, chans in rois.items():
            idx = [channels.index(c) for c in chans if c in channels]
            w, v = _pc1(data, idx)
            out[pid][roi] = (idx, w, v)
    return out


def equal_weights(channels, rois):
    """Weights that reproduce the channel average (for the wiring check)."""
    out = {}
    for roi, chans in rois.items():
        idx = [channels.index(c) for c in chans if c in channels]
        out[roi] = (idx, np.ones(len(idx)) / np.sqrt(len(idx)), np.nan)
    return out


def roi_series_pca(raw, channels, pid_weights):
    """Same output as whole_session_ddtf.roi_series: (n_regions x n_times array, region names)."""
    data = raw.copy().pick(channels).get_data()
    rows, names = [], []
    for roi, (idx, w, _) in pid_weights.items():
        rows.append(w @ data[idx])
        names.append(roi)
    return np.asarray(rows), names
