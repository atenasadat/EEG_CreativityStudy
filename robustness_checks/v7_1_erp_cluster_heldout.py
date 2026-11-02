"""Option A: pre-aha ERP cluster (analysis 1) and per-person + held-out (analysis 2).

Data: the cleaned files (EEG_DATA), with the 60 Hz notch on the 6 line-noise files (notch_patch).
Older note on the data mix:
  - 2 files postcleaning6 (notch before ICA, same manual rejection)
  - 5 files postcleaning1 with 60 Hz notch applied at read-time (notch_patch_optA)
  - 69 files postcleaning1 unchanged
Statistics mirror 03_erp_cluster.py and 26_loso_heldout_filtered.py.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os, sys  # noqa: E402
import json
import os
import numpy as np
import pandas as pd
import scipy.stats as st
import mne
from mne.stats import permutation_cluster_1samp_test

# notch patch MUST be imported before any mne.io.read_raw_eeglab call
import notch_patch  # noqa: F401
from common import *

mne.set_log_level('ERROR')
np.random.seed(0)
O = os.path.join(OUT, 'meeting_1013')
os.makedirs(O, exist_ok=True)
ROI = ['F3', 'C3', 'P3']
WIN = (-0.4501, -0.2966)
ALL15 = ['P4', 'P5', 'P6', 'P7', 'P9', 'P10', 'P11', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P20', 'P22']


def point_boundaries(raw):
    a = raw.annotations
    dur = [0.0 if str(d) == 'boundary' else float(x) for d, x in zip(a.description, a.duration)]
    raw.set_annotations(mne.Annotations(a.onset, dur, a.description, orig_time=a.orig_time))
    return raw


ep, ep_info = {}, []
times = info21 = None
for r in list_recordings():
    raw = read_raw(r['path'], preload=True).pick(STANDARD_CHANS)
    raw = point_boundaries(raw)
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    if '100' not in event_id:
        continue
    e = mne.Epochs(raw, events, event_id['100'], tmin=-0.5, tmax=0.5, baseline=(-0.2, 0),
                   preload=True, verbose='ERROR')
    if times is None:
        times, info21 = e.times.copy(), e.info
    ep.setdefault((r['pid'], r['cond']), []).append(e.get_data())
    ep_info.append(dict(fname=r['fname'], pid=r['pid'], cond=r['cond'], task=r['task'],
                        n_events100=int((events[:, 2] == event_id['100']).sum()), n_epochs=len(e)))
    print('loaded', r['fname'], flush=True)

ep_df = pd.DataFrame(ep_info)
per = ep_df.groupby(['pid', 'cond']).n_epochs.sum().unstack(fill_value=0).reindex(PARTICIPANTS, fill_value=0)
pids = [p for p in ALL15 if per.loc[p, 'control'] > 0 and per.loc[p, 'treatment'] > 0]
print('N =', len(pids), pids, flush=True)

C = np.stack([np.concatenate(ep[(p, 'control')], 0).mean(0) for p in pids]) * 1e6   # uV
T = np.stack([np.concatenate(ep[(p, 'treatment')], 0).mean(0) for p in pids]) * 1e6
np.savez(os.path.join(O, 'optB_participant_erps.npz'), control=C, treatment=T, times=times,
         chans=np.array(STANDARD_CHANS), pids=np.array(pids))

scalp_idx = [STANDARD_CHANS.index(c) for c in SCALP_CHANS]
info19 = mne.pick_info(info21, scalp_idx)
adj19, _ = mne.channels.find_ch_adjacency(info19, 'eeg')
roi_i_21 = [STANDARD_CHANS.index(c) for c in ROI]
roi_i_19 = [SCALP_CHANS.index(c) for c in ROI]
win = (times >= WIN[0]) & (times <= WIN[1])
orig_mask = np.zeros((len(times), len(SCALP_CHANS)), bool)
orig_mask[np.ix_(win, roi_i_19)] = True

# --------- Analysis 1: group cluster test (19 ch scalp adjacency, as in paper Methods) ---------
thr = st.t.ppf(0.975, len(pids) - 1)


def cluster(Cx, Tx, adjacency):
    X = np.transpose(Cx[:, scalp_idx] - Tx[:, scalp_idx], (0, 2, 1)) * 1e-6
    _, cl, pv, _ = permutation_cluster_1samp_test(X, n_permutations=min(2 ** len(X), 32768),
                                                  threshold=thr, tail=0, adjacency=adjacency,
                                                  out_type='mask', seed=0, verbose='ERROR')
    best = dict(p=1.0, window='', channels='', overlap=0.0, t_mean=0.0, mass=0.0, n_samples=0)
    for m, pval in zip(cl, pv):
        ov = float((m & orig_mask).sum() / orig_mask.sum())
        if ov > best['overlap']:
            ti, ci = np.where(m)
            best = dict(p=float(pval),
                        window=f'{times[ti.min()]:.3f}..{times[ti.max()]:.3f}',
                        channels=' '.join(SCALP_CHANS[c] for c in np.unique(ci)),
                        overlap=ov, t_mean=0.0, mass=0.0, n_samples=int(m.sum()))
    all_clusters = []
    for m, pval in zip(cl, pv):
        ti, ci = np.where(m)
        all_clusters.append(dict(p=float(pval),
                                 window=f'{times[ti.min()]:.3f}..{times[ti.max()]:.3f}',
                                 channels=' '.join(SCALP_CHANS[c] for c in np.unique(ci)),
                                 n_samples=int(m.sum())))
    return best, all_clusters


lat_best, lat_all = cluster(C, T, None)
sca_best, sca_all = cluster(C, T, adj19)
print('cluster lattice (19ch):', lat_best, flush=True)
print('cluster scalp adjacency:', sca_best, flush=True)

# --------- Analysis 2a: per-person ROI diff in the paper window ---------
c_roi = C[:, roi_i_21][:, :, win].mean((1, 2))
t_roi = T[:, roi_i_21][:, :, win].mean((1, 2))
d = c_roi - t_roi
n_paper_dir = int((d < 0).sum())  # paper direction: LLM > Unassisted (d < 0)
# Wait: paper says Unassisted - LLM in that window. In the paper Fig 2 LLM is more positive, so U - LLM < 0.
# The old scripts use (d < 0) = LLM more positive = paper direction. Keep that.
per_person_rows = [dict(pid=p, U_uV=float(c_roi[i]), LLM_uV=float(t_roi[i]), diff_U_minus_LLM=float(d[i]),
                        paper_direction=bool(d[i] < 0),
                        n_U=int(sum(len(x) for x in ep[(p, 'control')])),
                        n_LLM=int(sum(len(x) for x in ep[(p, 'treatment')])))
                   for i, p in enumerate(pids)]

ana1 = dict(
    N=len(pids), pids=pids,
    epochs_total=dict(unassisted=int(per.loc[pids, 'control'].sum()),
                      llm=int(per.loc[pids, 'treatment'].sum())),
    cluster_19ch_scalp_adjacency=sca_best,
    cluster_19ch_lattice=lat_best,
    all_clusters_scalp=sca_all,
    all_clusters_lattice=lat_all,
    roi_window_s=[float(times[win][0]), float(times[win][-1])],
    unassisted_roi_mean_uV=float(c_roi.mean()),
    llm_roi_mean_uV=float(t_roi.mean()),
    diff_U_minus_LLM_mean_uV=float(d.mean()),
    cohens_dz=float(d.mean() / d.std(ddof=1)),
    n_paper_direction_LLM_more_positive=n_paper_dir,
)
ana2a = dict(
    N=len(pids),
    n_paper_direction=n_paper_dir,
    sign_test_p=float(st.binomtest(n_paper_dir, len(pids)).pvalue),
    wilcoxon_p=float(st.wilcoxon(c_roi, t_roi).pvalue),
    mean_diff_U_minus_LLM_uV=float(d.mean()),
    median_diff_uV=float(np.median(d)),
    per_person=per_person_rows,
)
print('analysis 2a:', ana2a['n_paper_direction'], 'of', ana2a['N'],
      'sign_p=%.5f  wilcoxon_p=%.5f  median=%.2fuV' % (ana2a['sign_test_p'], ana2a['wilcoxon_p'], ana2a['median_diff_uV']),
      flush=True)

# --------- Analysis 2b: held-out leave-one-out window ---------
held, fold_rows = [], []
sc = [STANDARD_CHANS.index(c) for c in SCALP_CHANS]
for k, p in enumerate(pids):
    keep = [i for i in range(len(pids)) if i != k]
    X = np.transpose(C[keep][:, sc] - T[keep][:, sc], (0, 2, 1))
    thr_lo = st.t.ppf(0.975, len(keep) - 1)
    _, cl, pv, _ = permutation_cluster_1samp_test(X, n_permutations=2 ** len(keep), threshold=thr_lo, tail=0,
                                                  adjacency=None, out_type='mask', seed=0, verbose='ERROR')
    if len(cl):
        j = int(np.argmin(pv))
        ti, ci = np.where(cl[j])
        tm = np.zeros(len(times), bool); tm[np.unique(ti)] = True
        chs = [SCALP_CHANS[c] for c in np.unique(ci)]
        cp = float(pv[j])
    else:
        tm = (times >= WIN[0]) & (times <= WIN[1]); chs = ROI; cp = float('nan')
    ci_ = [STANDARD_CHANS.index(c) for c in chs]
    diff = float(C[k][ci_][:, tm].mean() - T[k][ci_][:, tm].mean())
    held.append(diff)
    fold_rows.append(dict(left_out=p, cluster_p_other14=cp, heldout_diff_uV=diff,
                          window=f'{times[tm].min():.3f}..{times[tm].max():.3f}' if tm.any() else '',
                          channels=' '.join(chs)))
held = np.array(held)
n_pd = int((held < 0).sum())
ana2b = dict(N=len(held), n_paper_direction=n_pd,
             sign_test_p=float(st.binomtest(n_pd, len(held)).pvalue),
             wilcoxon_p=float(st.wilcoxon(held).pvalue),
             median_uV=float(np.median(held)),
             folds=fold_rows)
print('analysis 2b (held-out):', n_pd, 'of', len(held),
      'sign_p=%.5f  wilcoxon_p=%.5f  median=%.2fuV' % (ana2b['sign_test_p'], ana2b['wilcoxon_p'], ana2b['median_uV']),
      flush=True)

json.dump(ana1, open(os.path.join(O, 'optB_1_erp.json'), 'w'), indent=1)
json.dump(dict(per_person=ana2a, held_out=ana2b), open(os.path.join(O, 'optB_2_perperson_heldout.json'), 'w'), indent=1)
ep_df.to_csv(os.path.join(O, 'optB_epochs_per_recording.csv'), index=False)
print('wrote optB_1_erp.json, optB_2_perperson_heldout.json, optB_participant_erps.npz')
