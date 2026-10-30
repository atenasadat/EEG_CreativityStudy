"""Task 7: non-insight control, following v04_work/PLAN.md "Non-insight control - analysis spec" (fixed).

Non-insight epochs: random time-zero points in the same cleaned recordings of the N = 15 sample.
  * excluded: time zero within +/-5 s of any event 100, any event 3, any event 4 (3/4 bursts collapsed first,
    events of the same code < 0.5 s apart = one event);
  * the epoch (-0.5..+0.5 s, 301 samples) may not contain a 'boundary' event nor run past a recording edge;
  * epoch definition identical to the aha ERP: -0.5..+0.5 s, baseline -0.2..0 s, 21 channels (repo order).
Count matching: in every recording, the number of random epochs = the number of aha epochs in that recording
  (so per participant x condition the count equals that participant's aha count in that condition; task is
  matched too). Positions drawn uniformly from valid samples, without replacement within a draw.
1,000 draws, numpy default_rng(seed=20261006). Primary per-participant non-insight ERP = average over draws.
ROI = F3/C3/P3, window -450..-297 ms (samples -135..-89 at 300 Hz), not re-selected.
Tests:
  C-a Wilcoxon (two-sided, N = 15) Unassisted vs LLM on the primary non-insight ROI values.
  C-b per draw, the same Wilcoxon; % of draws with p < .05 (and % with p < .05 in the aha direction).
  C-d aha ERP vs primary non-insight ERP within each condition: whole-scalp spatio-temporal cluster permutation
      (paired, two-tailed, thr t(.975, 14), exhaustive sign flips). Reported with (M) MNE channel adjacency on 19
      scalp channels [Methods as written] and (R) the repo cell-14 implementation (adjacency=None, 21 channels).
Outputs: outputs/07_noninsight_results.json, 07_noninsight_draws.csv, 07_noninsight_erps.npz, 07_noninsight.log
"""
import json
import numpy as np
import pandas as pd
import scipy.stats as st
import mne
from mne.stats import permutation_cluster_1samp_test
from common import *

mne.set_log_level('ERROR')
SEED = 20261006
NDRAW = 1000
rng = np.random.default_rng(SEED)

z = np.load(f'{OUT}/03_participant_erps.npz')
AHA_C, AHA_T, times = z['control'], z['treatment'], z['times']
sample = list(z['pids'])
N = len(sample)
ep_rec = pd.read_csv(f'{OUT}/03_epochs_per_recording.csv')
nt = len(times); half = (nt - 1) // 2          # 150 samples
base = (times >= -0.2 - 1e-9) & (times <= 0 + 1e-9)
win = (times >= -0.4501) & (times <= -0.2966)
roi = [STANDARD_CHANS.index(c) for c in ['F3', 'C3', 'P3']]

recs = [r for r in list_recordings() if r['pid'] in sample]
pcs = [(p, c) for p in sample for c in ['control', 'treatment']]
sum_erp = {pc: np.zeros((21, nt)) for pc in pcs}
roi_draw = {pc: np.zeros(NDRAW) for pc in pcs}
rec_meta = []
# accumulate per participant x condition the per-draw sum of epochs (to average later)
acc = {pc: np.zeros((NDRAW, 21, nt), dtype=np.float32) for pc in pcs}
cnt = {pc: 0 for pc in pcs}
for r in recs:
    k = int(ep_rec[(ep_rec.pid == r['pid']) & (ep_rec.cond == r['cond']) & (ep_rec.task == r['task'])].n_epochs.sum())
    raw = read_raw(r['path'], preload=True)
    raw.pick(STANDARD_CHANS)
    sf = raw.info['sfreq']; X = raw.get_data(); n = X.shape[1]
    ann = annotations_by_code(raw)
    ex = np.r_[np.array([x[0] for x in ann.get('100', [])]),
               collapse_bursts([x[0] for x in ann.get('3', [])]),
               collapse_bursts([x[0] for x in ann.get('4', [])])]
    s = np.arange(n)
    valid = (s - half >= 0) & (s + half <= n - 1)
    tsec = s / sf
    for e in ex:
        valid &= np.abs(tsec - e) > 5.0
    b_on, _ = boundary_times(raw)
    for b in b_on:
        bs = b * sf  # boundary sits between samples (EEGLAB latency x.5)
        valid &= ~((s - half <= bs) & (s + half >= bs))
    vidx = np.where(valid)[0]
    rec_meta.append(dict(pid=r['pid'], cond=r['cond'], task=r['task'], k=k, n_samples=n,
                         valid_s=len(vidx) / sf, n_excl_events=len(ex)))
    if k == 0:
        continue
    assert len(vidx) >= k
    offs = np.arange(-half, half + 1)
    pc = (r['pid'], r['cond'])
    for d in range(NDRAW):
        pos = rng.choice(vidx, size=k, replace=False)
        ep = X[:, pos[:, None] + offs[None, :]]           # (21, k, nt)
        ep = ep - ep[:, :, base].mean(2, keepdims=True)
        acc[pc][d] += ep.sum(1).astype(np.float32)
    cnt[pc] += k
    print(r['fname'], 'k =', k, 'valid min =', round(len(vidx) / sf / 60, 2), flush=True)

pd.DataFrame(rec_meta).to_csv(f'{OUT}/07_noninsight_recordings.csv', index=False)
RND = {pc: acc[pc] / cnt[pc] for pc in pcs}                   # (NDRAW, 21, nt) per-draw ERP
RC = np.stack([RND[(p, 'control')].mean(0) for p in sample])  # primary non-insight ERPs
RT = np.stack([RND[(p, 'treatment')].mean(0) for p in sample])
roiv = lambda A: A[..., roi, :][..., win].mean((-1, -2)) * 1e6
rc_draw = np.stack([roiv(RND[(p, 'control')]) for p in sample], 1)    # (NDRAW, N)
rt_draw = np.stack([roiv(RND[(p, 'treatment')]) for p in sample], 1)
np.savez(f'{OUT}/07_noninsight_erps.npz', control=RC, treatment=RT, times=times, pids=np.array(sample),
         chans=np.array(STANDARD_CHANS), roi_control_draws=rc_draw, roi_treatment_draws=rt_draw)

res = dict(seed=SEED, n_draws=NDRAW, N=N, epochs_per_condition={c: int(sum(cnt[(p, c)] for p in sample)) for c in ['control', 'treatment']})

# ---- C-a
rc, rt = roiv(RC), roiv(RT)
w = st.wilcoxon(rc, rt); d = rc - rt
res['C_a'] = dict(unassisted_mean_uV=float(rc.mean()), unassisted_sd=float(rc.std(ddof=1)),
                  llm_mean_uV=float(rt.mean()), llm_sd=float(rt.std(ddof=1)),
                  diff_U_minus_LLM_mean=float(d.mean()), diff_sd=float(d.std(ddof=1)),
                  dz_U_minus_LLM=float(d.mean() / d.std(ddof=1)), W=float(w.statistic), p=float(w.pvalue),
                  n_llm_more_positive=int((d < 0).sum()),
                  paired_t=float(st.ttest_rel(rc, rt).statistic), paired_t_p=float(st.ttest_rel(rc, rt).pvalue))
ac, at = roiv(AHA_C), roiv(AHA_T)
res['aha_reference'] = dict(unassisted_mean_uV=float(ac.mean()), llm_mean_uV=float(at.mean()),
                            W=float(st.wilcoxon(ac, at).statistic), p=float(st.wilcoxon(ac, at).pvalue),
                            dz=float((ac - at).mean() / (ac - at).std(ddof=1)))
# not pre-specified, labelled as such: interaction (aha difference minus non-insight difference) at the same ROI
inter = (ac - at) - (rc - rt)
wi = st.wilcoxon(inter)
res['extra_not_prespecified_interaction'] = dict(mean=float(inter.mean()), W=float(wi.statistic), p=float(wi.pvalue))
# aha vs non-insight at ROI within condition (descriptive, same ROI/window)
for lab, a_, r_ in [('control', ac, rc), ('treatment', at, rt)]:
    ww = st.wilcoxon(a_, r_)
    res[f'extra_roi_aha_vs_noninsight_{lab}'] = dict(aha_mean=float(a_.mean()), noninsight_mean=float(r_.mean()),
                                                     W=float(ww.statistic), p=float(ww.pvalue))

# ---- C-b
pd_rows = []
for dd in range(NDRAW):
    ww = st.wilcoxon(rc_draw[dd], rt_draw[dd])
    diff = rc_draw[dd] - rt_draw[dd]
    pd_rows.append(dict(draw=dd, W=float(ww.statistic), p=float(ww.pvalue), mean_diff_U_minus_LLM=float(diff.mean()),
                        median_diff=float(np.median(diff))))
dr = pd.DataFrame(pd_rows); dr.to_csv(f'{OUT}/07_noninsight_draws.csv', index=False)
res['C_b'] = dict(pct_p_lt_05=float((dr.p < 0.05).mean() * 100),
                  pct_p_lt_05_aha_direction=float(((dr.p < 0.05) & (dr.median_diff < 0)).mean() * 100),
                  pct_p_lt_05_opposite_direction=float(((dr.p < 0.05) & (dr.median_diff > 0)).mean() * 100),
                  median_p=float(dr.p.median()), p_q025_q975=[float(dr.p.quantile(.025)), float(dr.p.quantile(.975))],
                  median_W=float(dr.W.median()),
                  mean_diff_U_minus_LLM_median=float(dr.mean_diff_U_minus_LLM.median()),
                  mean_diff_q025_q975=[float(dr.mean_diff_U_minus_LLM.quantile(.025)), float(dr.mean_diff_U_minus_LLM.quantile(.975))],
                  aha_mean_diff=float((ac - at).mean()),
                  pct_draws_mean_diff_as_negative_as_aha=float((dr.mean_diff_U_minus_LLM <= (ac - at).mean()).mean() * 100))

# ---- C-d
info = mne.create_info(SCALP_CHANS, 300.0, 'eeg')
raw0 = read_raw(recs[0]['path'], preload=False).pick(SCALP_CHANS)
adj19, _ = mne.channels.find_ch_adjacency(raw0.info, 'eeg')
scalp_idx = [STANDARD_CHANS.index(c) for c in SCALP_CHANS]
thr = st.t.ppf(0.975, N - 1)

def clus(A, R, chans_idx, adjacency, label):
    Xd = np.transpose(A - R, (0, 2, 1))[:, :, chans_idx]
    names = [STANDARD_CHANS[i] for i in chans_idx]
    t_obs, cl, pv, H0 = permutation_cluster_1samp_test(Xd, n_permutations=32768, threshold=thr, tail=0,
                                                       adjacency=adjacency, out_type='mask', seed=0, verbose='ERROR')
    out = []
    for i in np.argsort(pv)[:5]:
        ti, ci = np.where(cl[i])
        out.append(dict(p=float(pv[i]), t_start=float(times[ti.min()]), t_end=float(times[ti.max()]),
                        channels=[names[c] for c in np.unique(ci)], mean_T=float(t_obs[cl[i]].mean()),
                        sign='aha > non-insight' if t_obs[cl[i]].mean() > 0 else 'aha < non-insight'))
    print(label, [(round(o['p'], 4), round(o['t_start'], 3), round(o['t_end'], 3), o['channels'], round(o['mean_T'], 2)) for o in out])
    return dict(n_clusters=len(cl), top=out)

res['C_d'] = {}
for c, A, R in [('Unassisted', AHA_C, RC), ('LLM-assisted', AHA_T, RT)]:
    res['C_d'][c] = dict(M_mne_adjacency_19ch=clus(A, R, scalp_idx, adj19, f'{c} M'),
                         R_repo_lattice_21ch=clus(A, R, list(range(21)), None, f'{c} R'))
json.dump(res, open(f'{OUT}/07_noninsight_results.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'C_d'}, indent=1))
