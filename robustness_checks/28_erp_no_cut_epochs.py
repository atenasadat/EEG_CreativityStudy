"""Pre-insight ERP leaving out every aha epoch that crosses a removed-data cut (boundary inside -0.5..+0.5 s).
Same epoching and cluster test as the paper. Data folder from EEG_DATA. Output prefix from OUT_PREFIX.
Saves <prefix>_participant_erps_nocut.npz (for 26_loso_heldout_filtered.py) and <prefix>_nocut_results.json."""
import json
import os
import numpy as np
import scipy.stats as st
import mne
from mne.stats import permutation_cluster_1samp_test
from common import *

mne.set_log_level('ERROR')
PREFIX = os.environ.get('OUT_PREFIX', 'nocut')
O = os.path.join(OUT, 'meeting_1013')
ROI, WIN = ['F3', 'C3', 'P3'], (-0.4501, -0.2966)
ep, n_drop = {}, 0
times = None
for r in list_recordings():
    raw = read_raw(r['path'], preload=True).pick(STANDARD_CHANS)
    events, eid = mne.events_from_annotations(raw)
    if '100' not in eid:
        continue
    tb = np.array([o for o, d in zip(raw.annotations.onset, raw.annotations.description) if d == 'boundary'])
    e = mne.Epochs(raw, events, eid['100'], tmin=-0.5, tmax=0.5, baseline=(-0.2, 0), preload=True)
    times = e.times
    t100 = e.events[:, 0] / raw.info['sfreq']
    keep = np.array([not (len(tb) and np.any((tb > t - 0.5) & (tb < t + 0.5))) for t in t100])
    n_drop += int((~keep).sum())
    ep.setdefault((r['pid'], r['cond']), []).append(e.get_data()[keep] * 1e6)
ALL15 = ['P4', 'P5', 'P6', 'P7', 'P9', 'P10', 'P11', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P20', 'P22']
pids = [p for p in ALL15 if sum(len(x) for x in ep.get((p, 'control'), [])) and sum(len(x) for x in ep.get((p, 'treatment'), []))]
C = np.stack([np.concatenate(ep[(p, 'control')]).mean(0) for p in pids])
T = np.stack([np.concatenate(ep[(p, 'treatment')]).mean(0) for p in pids])
np.savez(os.path.join(O, f'{PREFIX}_participant_erps_nocut.npz'), control=C, treatment=T, times=times,
         chans=np.array(STANDARD_CHANS), pids=np.array(pids))
sc = [STANDARD_CHANS.index(c) for c in SCALP_CHANS]
win = (times >= WIN[0]) & (times <= WIN[1])
mask0 = np.zeros((len(times), 19), bool)
mask0[np.ix_(win, [SCALP_CHANS.index(c) for c in ROI])] = True
X = np.transpose(C[:, sc] - T[:, sc], (0, 2, 1))
_, cl, pv, _ = permutation_cluster_1samp_test(X, n_permutations=min(2 ** len(X), 32768), threshold=st.t.ppf(0.975, len(X) - 1),
                                              tail=0, adjacency=None, out_type='mask', seed=0)
best = {}
for m, p in zip(cl, pv):
    ov = (m & mask0).sum() / mask0.sum()
    if ov > best.get('overlap', 0):
        ti, ci = np.where(m)
        best = dict(p=float(p), overlap=float(ov), window=f'{times[ti.min()]:.3f}..{times[ti.max()]:.3f}',
                    channels=' '.join(SCALP_CHANS[c] for c in np.unique(ci)))
ri = [STANDARD_CHANS.index(c) for c in ROI]
d = C[:, ri][:, :, win].mean((1, 2)) - T[:, ri][:, :, win].mean((1, 2))
res = dict(N=len(pids), dropped_epochs=n_drop,
           epochs=dict(control=int(sum(sum(len(x) for x in ep[(p, 'control')]) for p in pids)),
                       treatment=int(sum(sum(len(x) for x in ep[(p, 'treatment')]) for p in pids))),
           cluster_lattice=best, n_paper_direction=int((d < 0).sum()), wilcoxon_p=float(st.wilcoxon(d).pvalue),
           mean_diff_uV=float(d.mean()), values=dict(zip(pids, np.round(d, 2).tolist())))
json.dump(res, open(os.path.join(O, f'{PREFIX}_nocut_results.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
