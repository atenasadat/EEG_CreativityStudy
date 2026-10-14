import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
import numpy as np
import pandas as pd


BASE_PATH = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
BASE_PATH_ICA = "/Users/athenasaghi/Desktop/CleanDATA/CLEAN_BASELINES/ICA/"
PROJECT_ROOT = "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/"
TRIGGER_ID = "100"

INTERPOLATED_BASELINE_FILES = {
    "P10": "P10_baseline_ICA_interpolated.set",
    "P15": "P15_baseline_ICA_interpolated.set",
    "P20": "P20_baseline_ICA_interpolated.set",
    "P23": "P23_baseline_ICA_interpolated.set",
}

STANDARD_CHANNELS = [
    "P3", "C3", "F3", "Fz", "F4", "C4", "P4", "Cz", "Fp1",
    "Fp2", "T7", "P7", "O1", "O2", "F7", "F8", "P8", "T8", "Pz",
]

ALPHA_FREQS = np.arange(8, 14, 1.0);  ALPHA_CYC = 2.0
GAMMA_FREQS = np.arange(35, 51, 2.0); GAMMA_CYC = 4.0

MARKER_SPECS = {
    "Cluster51": dict(kind="amp", chans=["P3", "C3", "F3"], win=(-0.450, -0.297),
                      freqs=None, cyc=None, title="Cluster 51 ERP"),
    "Alpha": dict(kind="tfr", chans=["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8"],
                  win=(-0.5, 0.0), freqs=ALPHA_FREQS, cyc=ALPHA_CYC,
                  title="Frontal Alpha"),
    "Gamma": dict(kind="tfr", chans=["T8", "P8"], win=(-0.3, 0.0),
                  freqs=GAMMA_FREQS, cyc=GAMMA_CYC, title="Right Temporal Gamma"),
    "PrefrontalGamma": dict(kind="tfr", chans=["Fp1", "Fp2", "F3", "Fz", "F4"],
                            win=(-0.3, 0.0), freqs=GAMMA_FREQS, cyc=GAMMA_CYC,
                            title="Prefrontal Gamma"),
}


def condition_from_filename(fname):
    lower = fname.lower()
    if "control" in lower:
        return "control"
    if "treatment" in lower:
        return "treatment"
    return None


def normalize_pid(pid_values):
    return pid_values.astype(str).str.replace(" ", "", regex=False)


def resolve_eeg_path(fname, base_path=BASE_PATH):
    fname = os.fspath(fname)
    if os.path.isabs(fname):
        return fname
    return os.path.join(base_path, fname)


def read_eeg_file(fname, base_path=BASE_PATH, preload=True, channels=None, verbose=False):
    raw = mne.io.read_raw_eeglab(
        resolve_eeg_path(fname, base_path=base_path),
        preload=preload,
        verbose=verbose,
    )
    if channels is not None:
        raw.pick(channels)
    return raw


def get_baseline_file_for_pid(pid, base_path_ica=BASE_PATH_ICA):
    candidates = [
        INTERPOLATED_BASELINE_FILES.get(pid),
        f"{pid}_baseline_ICA_interpolated.set",
        f"{pid}_baseline_ICA.set",
    ]
    for fname in candidates:
        if fname and os.path.exists(os.path.join(base_path_ica, fname)):
            return fname
    return None


def load_resting_baseline_epochs(
    pid,
    base_path_ica=BASE_PATH_ICA,
    channels=None,
    epoch_duration=2.0,
):
    fname = get_baseline_file_for_pid(pid, base_path_ica=base_path_ica)
    if fname is None:
        return None, None

    raw = read_eeg_file(
        fname,
        base_path=base_path_ica,
        preload=True,
        channels=channels,
    )
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=epoch_duration,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs, fname


def resting_tfr_log_band_power(
    pid,
    channels,
    freqs,
    cyc_div,
    ch_names=None,
    base_path_ica=BASE_PATH_ICA,
    epoch_duration=2.0,
    decim=4,
):
    epochs, fname = load_resting_baseline_epochs(
        pid,
        base_path_ica=base_path_ica,
        channels=channels,
        epoch_duration=epoch_duration,
    )
    if epochs is None or not len(epochs):
        return None, fname, 0

    freqs = np.asarray(freqs, float)
    tfr = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=freqs / cyc_div,
        return_itc=False,
        average=False,
        decim=decim,
        verbose=False,
    )
    power = tfr.data.mean(axis=(0, 2, 3))
    log_power = np.log10(np.maximum(power, np.finfo(float).tiny))

    if ch_names is None:
        return log_power, fname, len(epochs)

    by_channel = dict(zip(tfr.ch_names, log_power))
    missing = [ch for ch in ch_names if ch not in by_channel]
    if missing:
        raise ValueError(f"Missing baseline channels for {pid}: {missing}")
    return np.asarray([by_channel[ch] for ch in ch_names]), fname, len(epochs)


def list_eeg_files(base_path=BASE_PATH, prefix_tag="postcleaning"):
    files = []
    for fname in sorted(os.listdir(base_path)):
        if not fname.endswith(".set") or prefix_tag not in fname:
            continue
        if condition_from_filename(fname):
            files.append(fname)
    return files


def get_condition_files(base_path=BASE_PATH, prefix_tag="postcleaning"):
    control_files, treatment_files = [], []
    for fname in list_eeg_files(base_path, prefix_tag=prefix_tag):
        if condition_from_filename(fname) == "control":
            control_files.append(fname)
        elif condition_from_filename(fname) == "treatment":
            treatment_files.append(fname)
    return control_files, treatment_files


def get_condition_file_groups(base_path=BASE_PATH, prefix_tag="postcleaning"):
    control_files, treatment_files = get_condition_files(base_path, prefix_tag=prefix_tag)
    return match_participant_file_groups(control_files, treatment_files)


def get_files_by_prefix(base_path=BASE_PATH, prefix_tag="postcleaning"):
    return get_condition_files(base_path, prefix_tag=prefix_tag)


def match_participants(control_files, treatment_files):
    c_dict = {f.split("_")[0]: f for f in control_files}
    t_dict = {f.split("_")[0]: f for f in treatment_files}
    matched = sorted(set(c_dict) & set(t_dict), key=lambda x: int(x.replace("P", "")))
    return matched, c_dict, t_dict


def get_files_with_trigger(base_path=BASE_PATH, trigger_id=TRIGGER_ID, verbose=False):
    control_files, treatment_files = [], []
    for f in list_eeg_files(base_path):
        raw = read_eeg_file(f, base_path=base_path, preload=False)
        _, event_id = mne.events_from_annotations(raw, verbose=False)
        if trigger_id in event_id:
            condition = condition_from_filename(f)
        else:
            if verbose:
                print(f"✗ Trigger '{trigger_id}' not in {f}. Available: {list(event_id.keys())}")
            continue
        if condition == "control":
            control_files.append(f)
        elif condition == "treatment":
            treatment_files.append(f)
    if verbose:
        print(f"Found {len(control_files)} control and {len(treatment_files)} treatment files with trigger '{trigger_id}'.")
        print(control_files)
        print(treatment_files)
    return control_files, treatment_files


def get_trigger_file_groups(base_path=BASE_PATH, trigger_id=TRIGGER_ID, verbose=False):
    control_files, treatment_files = get_files_with_trigger(
        base_path=base_path,
        trigger_id=trigger_id,
        verbose=verbose,
    )
    return match_participant_file_groups(control_files, treatment_files)


def group_files_by_participant(files):
    grouped = {}
    for f in sorted(files):
        grouped.setdefault(f.split("_")[0], []).append(f)
    return grouped


def match_participant_file_groups(control_files, treatment_files):
    c_groups = group_files_by_participant(control_files)
    t_groups = group_files_by_participant(treatment_files)
    matched = sorted(set(c_groups) & set(t_groups), key=lambda x: int(x.replace("P", "")))
    return matched, c_groups, t_groups


def load_aha_epochs_from_files(
    files,
    base_path=BASE_PATH,
    trigger_id=TRIGGER_ID,
    tmin=-2.0,
    tmax=1.0,
    baseline=(-0.2, 0),
):
    epoch_list = []
    for fname in files:
        raw = read_eeg_file(fname, base_path=base_path, preload=True, channels=STANDARD_CHANNELS)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        if trigger_id not in event_id:
            continue
        epochs = mne.Epochs(
            raw, events, event_id[trigger_id], tmin=tmin, tmax=tmax,
            baseline=baseline, preload=True, verbose=False,
        )
        if len(epochs):
            epoch_list.append(epochs)
    if not epoch_list:
        return None
    if len(epoch_list) == 1:
        return epoch_list[0]
    return mne.concatenate_epochs(epoch_list, verbose=False)


def load_raws_from_files(files, base_path=BASE_PATH, channels=None):
    raws = []
    for fname in files:
        raw = read_eeg_file(fname, base_path=base_path, preload=True, channels=channels)
        raws.append(raw)
    if not raws:
        return None
    if len(raws) == 1:
        return raws[0]
    return mne.concatenate_raws(raws, verbose=False)


def marker_epoch_bounds(marker_specs, default_tmin=-2.0, default_tmax=1.0):
    tmins = [default_tmin]
    tmaxs = [default_tmax]
    for spec in marker_specs.values():
        win = spec.get("win")
        if win is None:
            continue
        tmins.append(float(win[0]))
        tmaxs.append(float(win[1]))
    return min(tmins), max(tmaxs)


def load_behavior_scores(project_root=PROJECT_ROOT):
    dat_df = pd.read_csv(os.path.join(project_root, "DATScores.csv"))
    aut_df = pd.read_csv(os.path.join(project_root, "AUTScores.csv"))
    dat_df["Pid"] = normalize_pid(dat_df["Pid"])
    aut_df["Pid"] = normalize_pid(aut_df["Pid"])
    df = pd.merge(
        dat_df[["Pid", "dat"]],
        aut_df[["Pid", "Total", "Fluency", "Flexibility", "Elaboration", "Originality"]],
        on="Pid",
        how="inner",
    )
    return df.rename(columns={"dat": "DAT_Score", "Total": "AUT_Total"})


def _mean(a):
    return float(np.mean(a)) if len(a) else np.nan


def amp_trials(epochs, channels, tmin, tmax):
    ep = epochs.copy().pick(channels).crop(tmin, tmax)
    data = ep.get_data()
    return data.mean(axis=(1, 2)) * 1e6 if data.size else np.array([])


def tfr_trials(epochs, channels, tmin, tmax, freqs, cyc_div):
    ep = epochs.copy().pick(channels)
    tfr = ep.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=freqs / cyc_div,
        return_itc=False,
        average=False,
        verbose=False,
    )
    tfr.crop(tmin, tmax)
    return tfr.data.mean(axis=(1, 2, 3)) if tfr.data.size else np.array([])


def tfr_trials_cached(epochs, spec, cache):
    chans = tuple(spec["chans"])
    freqs = tuple(np.asarray(spec["freqs"], float))
    cache_key = (freqs, float(spec["cyc"]))

    if cache_key not in cache:
        ep = epochs.copy()
        cache[cache_key] = ep.compute_tfr(
            method="morlet",
            freqs=np.asarray(freqs),
            n_cycles=np.asarray(freqs) / spec["cyc"],
            return_itc=False,
            average=False,
            verbose=False,
        )

    tfr = cache[cache_key]
    picks = [tfr.ch_names.index(ch) for ch in chans if ch in tfr.ch_names]
    if not picks:
        return np.array([])

    tmin, tmax = spec["win"]
    mask = (tfr.times >= tmin) & (tfr.times <= tmax)
    if not mask.any():
        return np.array([])
    data = tfr.data[:, picks, :, :]
    return data[:, :, :, mask].mean(axis=(1, 2, 3))


def extract_erp_features(include_trials=False, base_path=BASE_PATH, marker_specs=None):
    marker_specs = marker_specs or MARKER_SPECS
    matched_pids, c_groups, t_groups = get_trigger_file_groups(base_path)
    epoch_tmin, epoch_tmax = marker_epoch_bounds(marker_specs)

    rows = []
    trial_long = {key: [] for key in marker_specs}
    trial_counts = []

    for pid in matched_pids:
        ep_c = load_aha_epochs_from_files(
            c_groups[pid], base_path=base_path, tmin=epoch_tmin, tmax=epoch_tmax
        )
        ep_t = load_aha_epochs_from_files(
            t_groups[pid], base_path=base_path, tmin=epoch_tmin, tmax=epoch_tmax
        )

        if ep_c is None or ep_t is None:
            print(f"Skipping {pid}: no aha epochs in both conditions.")
            continue

        row = {"Pid": pid}
        trial_counts.append({"Pid": pid, "n_control": len(ep_c), "n_treatment": len(ep_t)})
        tfr_cache_c, tfr_cache_t = {}, {}

        for key, spec in marker_specs.items():
            if spec["kind"] == "amp":
                tc = amp_trials(ep_c, spec["chans"], *spec["win"])
                tt = amp_trials(ep_t, spec["chans"], *spec["win"])
            else:
                tc = tfr_trials_cached(ep_c, spec, tfr_cache_c)
                tt = tfr_trials_cached(ep_t, spec, tfr_cache_t)

            row[f"{key}_Control"] = _mean(tc)
            row[f"{key}_Treatment"] = _mean(tt)
            row[f"{key}_Diff"] = row[f"{key}_Treatment"] - row[f"{key}_Control"]

            if include_trials:
                trial_long[key].extend({"subject": pid, "condition": 0, "amplitude": v} for v in tc)
                trial_long[key].extend({"subject": pid, "condition": 1, "amplitude": v} for v in tt)

        rows.append(row)
        print(f"-> {pid}: control={len(ep_c)} treatment={len(ep_t)} aha epochs")

    return pd.DataFrame(rows), trial_long, pd.DataFrame(trial_counts)


def build_erp_behavior_table(include_trials=False, marker_specs=None):
    df_eeg, trial_long, trial_counts = extract_erp_features(
        include_trials=include_trials,
        marker_specs=marker_specs,
    )
    df_final = pd.merge(load_behavior_scores(), df_eeg, on="Pid", how="inner")
    return df_final, trial_long, trial_counts
