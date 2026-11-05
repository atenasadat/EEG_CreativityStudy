"""Paper v08, pre-insight power. Power before the aha vs random non-insight moments from the same recordings (insight-literature design).
Morlet as Athena (1-49 Hz, 2-Hz steps, n_cycles f/2, decim 2). Epochs -5.5..5 s for aha (event 100) and for 40 random
time points per recording (> 5 s from events 100/3/4, no cut inside the epoch, inside the recording).
dB per epoch vs (a) the recording's mean power over the random epochs' -1..1 s ('session'), (b) the epoch's own -5..-4 s
('early'), (c) none (absolute). Saved: per person x condition x type (aha, random) mean maps for all 19 channels.
aha - random removes the log-averaging bias and any general condition difference in power.
No notch or other filter. Usage: EEG_DATA=<cleaned files> OUT=<folder> python v8_pre_aha_power.py, then
MAPS=<folder>/maps_v2.npz python v8_pre_aha_power_stats.py"""
import json, os, sys, warnings
import numpy as np, mne
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'ERP'))   # your ERP folder (Morlet settings, file lists)
import erp_shared
BASE = os.environ['EEG_DATA'].rstrip('/') + '/'
erp_shared.BASE_PATH = BASE
import erp_tfr_tf_cluster_pre_event as A
sys.path.insert(0, HERE)
from common import annotations_by_code, collapse_bursts, boundary_times
mne.set_log_level('ERROR'); warnings.filterwarnings('ignore')
OUT = os.environ['OUT']; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261104)
TMIN, TMAX = -5.5, 5.0
FREQS, NCYC, DECIM = A.FREQS, A.FREQS / A.CYCL_DIV, A.DECIM
CH = A.CHANNELS
K_RAND, KEEP = 40, 3
tiny = np.finfo(float).tiny

def tfr(d, sf):
    out = []
    for i in range(0, len(d), 10):
        out.append(mne.time_frequency.tfr_array_morlet(d[i:i + 10], sf, FREQS, n_cycles=NCYC, output='power', decim=DECIM, n_jobs=1))
    return np.concatenate(out)

pids, c_groups, t_groups = erp_shared.get_trigger_file_groups(BASE, verbose=False)
res, times = {}, None
for pid in pids:
    for cond, files in [('control', c_groups[pid]), ('treatment', t_groups[pid])]:
        acc = {f'{ty}|{b}': [] for ty in ['aha', 'random'] for b in ['session', 'early', 'none']}
        for f in files:
            raw = erp_shared.read_eeg_file(f, base_path=BASE, preload=True, channels=erp_shared.STANDARD_CHANNELS)
            sf = raw.info['sfreq']
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if erp_shared.TRIGGER_ID not in event_id:
                continue
            ep = mne.Epochs(raw, events, event_id[erp_shared.TRIGGER_ID], tmin=TMIN, tmax=TMAX, baseline=None, preload=True, verbose=False).pick(CH)
            if len(ep) == 0:
                continue
            X = raw.copy().pick(CH).get_data(); n = X.shape[1]
            ann = annotations_by_code(raw)
            ex = np.r_[np.array([x[0] for x in ann.get('100', [])]), collapse_bursts([x[0] for x in ann.get('3', [])]),
                       collapse_bursts([x[0] for x in ann.get('4', [])])]
            b_on, _ = boundary_times(raw)
            s = np.arange(n); tsec = s / sf
            a0, a1 = int(round(-TMIN * sf)), int(round(TMAX * sf))
            valid = (s - a0 >= 0) & (s + a1 < n)
            for e in ex: valid &= np.abs(tsec - e) > 5.0
            for b in b_on: valid &= ~((s - a0 <= b * sf) & (s + a1 >= b * sf))
            vidx = np.where(valid)[0]
            pos = rng.choice(vidx, size=min(K_RAND, len(vidx)), replace=False)
            R = np.stack([X[:, p - a0:p + a1 + 1] for p in pos])
            Ea = ep.get_data()
            m = min(R.shape[-1], Ea.shape[-1]); R, Ea = R[..., :m], Ea[..., :m]
            pa, pr = tfr(Ea, sf), tfr(R, sf)
            t = np.arange(pa.shape[-1]) * DECIM / sf + TMIN
            keep = np.arange(0, len(t), KEEP)
            if times is None: times = t[keep]
            sess = pr[..., (t >= -1.0) & (t <= 1.0)].mean(axis=(0, 3))[None, :, :, None]
            for ty, p in [('aha', pa), ('random', pr)]:
                early = p[..., (t >= -5.0) & (t <= -4.0)].mean(-1, keepdims=True)
                acc[f'{ty}|session'].append((10 * np.log10(np.maximum(p, tiny) / sess))[..., keep])
                acc[f'{ty}|early'].append((10 * np.log10(np.maximum(p, tiny) / early))[..., keep])
                acc[f'{ty}|none'].append((10 * np.log10(np.maximum(p, tiny) * 1e12))[..., keep])
        if not acc['aha|session']:
            continue
        for k, v in acc.items():
            v = np.concatenate(v)
            res[f'{pid}|{cond}|{k}'] = v.mean(0).astype(np.float32)
            res[f'{pid}|{cond}|{k}|n'] = np.array(len(v))
        print(pid, cond, 'aha epochs', int(res[f'{pid}|{cond}|aha|session|n']), 'random', int(res[f'{pid}|{cond}|random|session|n']), flush=True)
np.savez_compressed(f'{OUT}/maps_v2.npz', times=times, freqs=FREQS, chans=np.array(CH), **res)
print('DONE', flush=True)
