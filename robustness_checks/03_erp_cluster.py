"""Task 3: reproduce the N = 15 ERP cluster permutation test (repo ERP/ahaMoments_ERP.ipynb cell 14).

Epoching follows repo ERP/utils.py get_subject_erp exactly: mne.Epochs(raw, events, event_id['100'],
tmin=-0.5, tmax=0.5, baseline=(-0.2, 0), preload=True) on the 21 standard channels; all of a
participant's epochs in a condition concatenated and averaged (tasks merged).  Diff = control - treatment
(Unassisted - LLM), as in run_cluster_permutation_test.

Variants (all two-tailed, tail = 0; n_permutations = 32768 -> MNE uses the exact/exhaustive sign-flip
distribution because 32768 >= 2**(15-1)):
  R  repo cell 14 as run: 21 channels incl. A1/A2, adjacency=None (MNE lattice = neighbouring channel
     *indices* in file order are adjacent), default threshold t(0.975, 14).
  M  methods-as-described / task spec: 19 scalp channels (A1/A2 excluded), mne.channels.find_ch_adjacency,
     threshold t(0.975, 14) (alpha .05 two-tailed).
  M1 repo cell 16: as M but threshold t.ppf(0.95, 14) = 1.761 (the repo's one-tailed value used with tail=0).
  Each also run excluding aha epochs that span a 'boundary' event (sensitivity, 'nb').

Outputs: outputs/03_participant_erps.npz, outputs/03_erp_cluster_results.json, outputs/03_epochs_per_participant.csv
"""
import json
import numpy as np
import pandas as pd
import scipy.stats
import mne
from mne.stats import permutation_cluster_1samp_test
from common import *

mne.set_log_level('ERROR')
np.random.seed(0)

recs = list_recordings()
ep_info = []
erp = {}      # (pid, cond) -> (n_chan, n_times) average over all epochs
erp_nb = {}   # same, excluding boundary-spanning epochs
times = None
info21 = None
for r in recs:
    raw = read_raw(r['path'], preload=True)
    raw.pick(STANDARD_CHANS)
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    if '100' not in event_id:
        continue
    ep = mne.Epochs(raw, events, event_id['100'], tmin=-0.5, tmax=0.5, baseline=(-0.2, 0),
                    preload=True, verbose='ERROR')
    if times is None:
        times = ep.times.copy(); info21 = ep.info
    b_on, _ = boundary_times(raw)
    ev_t = ep.events[:, 0] / raw.info['sfreq']
    spans = np.array([np.any((b_on > t - 0.5) & (b_on < t + 0.5)) for t in ev_t], bool)
    key = (r['pid'], r['cond'])
    erp.setdefault(key, []).append(ep.get_data())
    erp_nb.setdefault(key, []).append(ep.get_data()[~spans])
    ep_info.append(dict(pid=r['pid'], cond=r['cond'], task=r['task'], n_events100=int((events[:, 2] == event_id['100']).sum()),
                        n_epochs=len(ep), n_span_boundary=int(spans.sum())))

ep_df = pd.DataFrame(ep_info)
ep_df.to_csv(f'{OUT}/03_epochs_per_recording.csv', index=False)
per = ep_df.groupby(['pid', 'cond']).n_epochs.sum().unstack(fill_value=0).reindex(PARTICIPANTS, fill_value=0)
per_nb = (ep_df.assign(n=ep_df.n_epochs - ep_df.n_span_boundary).groupby(['pid', 'cond']).n.sum()
          .unstack(fill_value=0).reindex(PARTICIPANTS, fill_value=0))
sample = [p for p in PARTICIPANTS if per.loc[p, 'control'] > 0 and per.loc[p, 'treatment'] > 0]
print('N =', len(sample), sample)
per.loc[sample].to_csv(f'{OUT}/03_epochs_per_participant.csv')

res = {'sample': sample, 'N': len(sample)}
res['epochs_N15'] = {c: dict(mean=float(per.loc[sample, c].mean()), sd=float(per.loc[sample, c].std(ddof=1)),
                             total=int(per.loc[sample, c].sum()), min=int(per.loc[sample, c].min()),
                             max=int(per.loc[sample, c].max())) for c in ['control', 'treatment']}
res['epochs_spanning_boundary_N15'] = {c: int((per.loc[sample, c] - per_nb.loc[sample, c]).sum()) for c in ['control', 'treatment']}
print(res['epochs_N15'], res['epochs_spanning_boundary_N15'])

def stack(d, pids, cond):
    return np.stack([np.concatenate(d[(p, cond)], 0).mean(0) for p in pids])  # (N, ch, t)

C = stack(erp, sample, 'control'); T = stack(erp, sample, 'treatment')
sample_nb = [p for p in sample if per_nb.loc[p, 'control'] > 0 and per_nb.loc[p, 'treatment'] > 0]
Cnb = stack(erp_nb, sample_nb, 'control'); Tnb = stack(erp_nb, sample_nb, 'treatment')
np.savez(f'{OUT}/03_participant_erps.npz', control=C, treatment=T, times=times, chans=np.array(STANDARD_CHANS),
         pids=np.array(sample), control_nb=Cnb, treatment_nb=Tnb, pids_nb=np.array(sample_nb))

scalp_idx = [STANDARD_CHANS.index(c) for c in SCALP_CHANS]
info19 = mne.pick_info(info21, scalp_idx)
adj19, adj_names = mne.channels.find_ch_adjacency(info19, 'eeg')
assert list(adj_names) == SCALP_CHANS
res['adjacency_neighbors'] = {SCALP_CHANS[i]: [SCALP_CHANS[j] for j in np.nonzero(adj19.toarray()[i])[0] if j != i]
                              for i in range(len(SCALP_CHANS))}


def run(Cx, Tx, chans_idx, adjacency, thr, label):
    n = Cx.shape[0]
    X = np.transpose(Cx - Tx, (0, 2, 1))[:, :, chans_idx]  # (N, times, ch)
    names = [STANDARD_CHANS[i] for i in chans_idx]
    t_obs, clusters, pv, H0 = permutation_cluster_1samp_test(
        X, n_permutations=32768, threshold=thr, tail=0, adjacency=adjacency, out_type='mask',
        seed=0, verbose='ERROR')
    out = dict(label=label, N=n, threshold=float(thr), n_clusters=len(clusters), n_H0=int(len(H0)), clusters=[])
    for i in np.argsort(pv):
        m = clusters[i]
        ti, ci = np.where(m)
        out['clusters'].append(dict(
            idx=int(i), p=float(pv[i]), t_start=float(times[ti.min()]), t_end=float(times[ti.max()]),
            channels=[names[c] for c in np.unique(ci)], mean_T=float(t_obs[m].mean()),
            mass=float(t_obs[m].sum()), n_samples=int(m.sum())))
    print(f'\n== {label}: N={n}, thr={thr:.4f}, n_clusters={len(clusters)}, len(H0)={len(H0)}')
    for c in out['clusters'][:6]:
        print(f"   p={c['p']:.5f}  {c['t_start']:.4f}..{c['t_end']:.4f}s  mean T={c['mean_T']:.3f}  mass={c['mass']:.2f}  ch={c['channels']}")
    return out

thr2 = scipy.stats.t.ppf(1 - 0.05 / 2, df=len(sample) - 1)
thr1 = scipy.stats.t.ppf(1 - 0.05, df=len(sample) - 1)
all21 = list(range(21))
res['R_repo_cell14_lattice_21ch'] = run(C, T, all21, None, thr2, 'R repo cell14 (lattice, 21 ch incl A1/A2)')
res['M_mne_adjacency_19ch'] = run(C, T, scalp_idx, adj19, thr2, 'M MNE adjacency, 19 scalp ch, t(.975)')
res['M1_mne_adjacency_19ch_thr1761'] = run(C, T, scalp_idx, adj19, thr1, 'M1 MNE adjacency, 19 ch, thr t(.95)=1.761')
res['R_lattice_19ch'] = run(C, T, scalp_idx, None, thr2, 'R19 lattice, 19 scalp ch (A1/A2 removed from index order)')
thr2nb = scipy.stats.t.ppf(1 - 0.05 / 2, df=len(sample_nb) - 1)
res['R_nb'] = run(Cnb, Tnb, all21, None, thr2nb, 'R no-boundary-span epochs')
res['M_nb'] = run(Cnb, Tnb, scalp_idx, adj19, thr2nb, 'M no-boundary-span epochs')

# ROI effect size: F3/C3/P3 x -450..-297 ms (samples -135..-89 at 300 Hz)
win = (times >= -0.4501) & (times <= -0.2966)
roi = [STANDARD_CHANS.index(c) for c in ['F3', 'C3', 'P3']]
c_roi = C[:, roi][:, :, win].mean((1, 2)) * 1e6
t_roi = T[:, roi][:, :, win].mean((1, 2)) * 1e6
d = c_roi - t_roi
w = scipy.stats.wilcoxon(c_roi, t_roi)
tt = scipy.stats.ttest_rel(c_roi, t_roi)
res['roi'] = dict(window_s=[float(times[win][0]), float(times[win][-1])], n_samples=int(win.sum()),
                  unassisted_mean_uV=float(c_roi.mean()), unassisted_sd=float(c_roi.std(ddof=1)),
                  llm_mean_uV=float(t_roi.mean()), llm_sd=float(t_roi.std(ddof=1)),
                  diff_U_minus_LLM_mean_uV=float(d.mean()), diff_sd=float(d.std(ddof=1)),
                  cohens_dz_U_minus_LLM=float(d.mean() / d.std(ddof=1)),
                  cohens_dz_ddof0=float(d.mean() / d.std(ddof=0)),
                  paired_t=float(tt.statistic), paired_t_p=float(tt.pvalue),
                  wilcoxon_W=float(w.statistic), wilcoxon_p=float(w.pvalue),
                  n_llm_more_positive=int((d < 0).sum()),
                  per_participant=dict(zip(sample, map(float, d))))
print('\nROI', json.dumps({k: v for k, v in res['roi'].items() if k != 'per_participant'}, indent=1))
json.dump(res, open(f'{OUT}/03_erp_cluster_results.json', 'w'), indent=1)
