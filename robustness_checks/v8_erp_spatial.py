"""Pre-aha ERP cluster test as in the paper (v08): spatial neighbours, search before the aha.

Same participant ERPs, threshold and sign flips as before (03_erp_cluster.py output, N = 15, paired t,
two-tailed threshold t(.975, 14), all 2^15 sign flips). What changed from v7:
  - neighbours: channels are neighbours if they share an edge in a Delaunay triangulation of the 2-D projected
    electrode positions (mne.channels.find_ch_adjacency; no template exists for the DSI-24, so MNE triangulates).
    v7 used adjacency=None, which in MNE means a regular lattice: channels next to each other in the data file
    count as neighbours (Cz-Fp1, O2-F7, ...), which has no spatial meaning.
  - search window: -0.5 to 0 s (before the aha), because the question is about the period before an insight.
    This was chosen after the full -0.5 to 0.5 s search gave p = .080, so both are reported.
Also reported: the cluster summary on the cluster's channels and time span, the leave-one-participant-out check
with the same test, no channel neighbours as a check, the old file-order version, and the notch check
(ERPS_NOTCH = participant ERPs from v7_1_erp_cluster_heldout.py run with NOTCH=1).
Input: OUT_DIR/03_participant_erps.npz. Output: OUT_DIR/v8_erp_spatial.json, v8_erp_neighbours.json
"""
import json
import os

import mne
import numpy as np
import scipy.sparse as sp
import scipy.stats as st
from mne.stats import permutation_cluster_1samp_test

from common import *

mne.set_log_level('ERROR')
z = np.load(f'{OUT}/03_participant_erps.npz')
times, chans, pids = z['times'], list(z['chans']), list(z['pids'])
sc = [chans.index(c) for c in SCALP_CHANS]
N = len(pids)
raw0 = read_raw(list_recordings()[0]['path'], preload=False).pick(SCALP_CHANS)
ADJ = {'spatial (Delaunay)': mne.channels.find_ch_adjacency(raw0.info, 'eeg')[0],
       'no neighbours': sp.csr_matrix((len(SCALP_CHANS), len(SCALP_CHANS))),
       'file order (v7)': None}
SPANS = {'pre -0.5..0': times <= 1e-9, 'full -0.5..0.5': np.ones(len(times), bool)}


def diffs(npz):
    C, T = npz['control'], npz['treatment']
    scale = 1e6 if np.abs(C).max() < 1e-2 else 1.0          # 03 saves volts, v7_1 saves microvolts
    return np.transpose(C[:, sc] - T[:, sc], (0, 2, 1)) * scale, C * scale, T * scale


def best_cluster(X, tmask, adjacency):
    n = len(X)
    _, cl, pv, _ = permutation_cluster_1samp_test(X[:, tmask], n_permutations=2 ** n, threshold=st.t.ppf(.975, n - 1),
                                                  tail=0, adjacency=adjacency, out_type='mask', seed=0, verbose='ERROR')
    if not len(pv):
        return None
    k = int(np.argmin(pv)); m = cl[k]
    ti, ci = np.where(m)
    tt = times[tmask]
    return dict(p=float(pv[k]), t_start=float(tt[ti.min()]), t_end=float(tt[ti.max()]),
                channels=[SCALP_CHANS[c] for c in np.unique(ci)], n_clusters=len(cl))


X, C, T = diffs(z)
res = {'N': N, 'pids': pids, 'tests': {}}
for an, A in ADJ.items():
    for sn, tm in SPANS.items():
        r = best_cluster(X, tm, A)
        res['tests'][f'{an} | {sn}'] = r
        print(f'{an:20s} {sn:15s}', r, flush=True)

main = res['tests']['spatial (Delaunay) | pre -0.5..0']
win = (times >= main['t_start'] - 1e-9) & (times <= main['t_end'] + 1e-9)
roi = [chans.index(c) for c in main['channels']]
cu, cl_ = C[:, roi][:, :, win].mean((1, 2)), T[:, roi][:, :, win].mean((1, 2))
d = cu - cl_
w = st.wilcoxon(cu, cl_)
res['cluster_summary'] = dict(channels=main['channels'], window_s=[main['t_start'], main['t_end']],
                              unassisted_mean_uV=float(cu.mean()), unassisted_sd=float(cu.std(ddof=1)),
                              llm_mean_uV=float(cl_.mean()), llm_sd=float(cl_.std(ddof=1)),
                              diff_U_minus_LLM_mean_uV=float(d.mean()), cohens_dz=float(d.mean() / d.std(ddof=1)),
                              n_llm_more_positive=int((d < 0).sum()), W=float(w.statistic), wilcoxon_p=float(w.pvalue))
print('cluster summary', res['cluster_summary'], flush=True)

# leave one participant out: find the cluster on the other 14 with the same test, score the left-out person on
# that cluster's channels x time span (the same summary as above)
held, folds = [], []
tm = SPANS['pre -0.5..0']
for k in range(N):
    keep = [i for i in range(N) if i != k]
    r = best_cluster(X[keep], tm, ADJ['spatial (Delaunay)'])
    wk = (times >= r['t_start'] - 1e-9) & (times <= r['t_end'] + 1e-9)
    rk = [chans.index(c) for c in r['channels']]
    held.append(float(C[k][rk][:, wk].mean() - T[k][rk][:, wk].mean()))
    folds.append(dict(left_out=pids[k], cluster_p_other14=r['p'], channels=r['channels'],
                      window=[r['t_start'], r['t_end']], heldout_diff_U_minus_LLM_uV=held[-1]))
held = np.array(held)
res['held_out'] = dict(n_llm_more_positive=int((held < 0).sum()), N=N, wilcoxon_p=float(st.wilcoxon(held).pvalue),
                       sign_p=float(st.binomtest(int((held < 0).sum()), N).pvalue), median_uV=float(np.median(held)),
                       folds=folds)
print('held-out', {k: v for k, v in res['held_out'].items() if k != 'folds'}, flush=True)

if os.environ.get('ERPS_NOTCH'):
    zn = np.load(os.environ['ERPS_NOTCH'])
    assert list(zn['pids']) == pids
    Xn, _, _ = diffs(zn)
    res['notch_check'] = best_cluster(Xn, tm, ADJ['spatial (Delaunay)'])
    print('notch check', res['notch_check'], flush=True)

A = ADJ['spatial (Delaunay)'].toarray()
nb = {c: [SCALP_CHANS[j] for j in np.nonzero(A[i])[0] if j != i] for i, c in enumerate(SCALP_CHANS)}
json.dump(dict(neighbours=nb, mean_per_channel=float(np.mean([len(v) for v in nb.values()]))),
          open(f'{OUT}/v8_erp_neighbours.json', 'w'), indent=1)
json.dump(res, open(f'{OUT}/v8_erp_spatial.json', 'w'), indent=1)
print('ROI_CH=' + ','.join(main['channels']))
