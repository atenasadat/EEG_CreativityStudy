"""Tasks 1 and 2: aha counts per participant x condition and time on task.

Input : outputs/01_recording_inventory.csv (from 01_inventory.py)
Output: outputs/02_aha_counts_per_participant.csv, outputs/02_time_on_task_per_participant.csv,
        outputs/02_time_per_task.csv, outputs/02_counts_time_summary.json
"""
import json
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from common import *

inv = pd.read_csv(f'{OUT}/01_recording_inventory.csv')
order = {p: i for i, p in enumerate(PARTICIPANTS)}
res = {}

# ---------------- Task 1: aha counts ----------------
cnt = inv.pivot_table(index='pid', columns='cond', values='n100', aggfunc='sum').reindex(PARTICIPANTS)
cnt['total'] = cnt.control + cnt.treatment
cnt['in_N15'] = (cnt.control > 0) & (cnt.treatment > 0)
cnt.to_csv(f'{OUT}/02_aha_counts_per_participant.csv')
print(cnt.to_string())
res['counts_all19'] = dict(control=int(cnt.control.sum()), treatment=int(cnt.treatment.sum()),
                           total=int(cnt.total.sum()), n_participants=int(len(cnt)))
n15 = cnt[cnt.in_N15]
res['counts_N15'] = dict(control=int(n15.control.sum()), treatment=int(n15.treatment.sum()),
                         total=int(n15.total.sum()), participants=list(n15.index))
res['zero_control'] = list(cnt.index[cnt.control == 0])
res['zero_treatment'] = list(cnt.index[cnt.treatment == 0])
res['zero_both'] = list(cnt.index[cnt.total == 0])
res['mean_total_per_participant_all19'] = float(cnt.total.mean())
res['mean_per_participant_per_condition'] = dict(
    all19_control=[float(cnt.control.mean()), float(cnt.control.std(ddof=1))],
    all19_treatment=[float(cnt.treatment.mean()), float(cnt.treatment.std(ddof=1))],
    N15_control=[float(n15.control.mean()), float(n15.control.std(ddof=1))],
    N15_treatment=[float(n15.treatment.mean()), float(n15.treatment.std(ddof=1))])

# candidate sources of "5.83 per participant"
cands = {}
for lab, d in [('all19', cnt), ('N15', n15)]:
    cands[f'{lab}_mean_total'] = float(d.total.mean())
    cands[f'{lab}_mean_of_condition_means'] = float((d.control.mean() + d.treatment.mean()) / 2)
    cands[f'{lab}_mean_per_recording'] = float(d.total.sum() / (4 * len(d)))
    cands[f'{lab}_median_total'] = float(d.total.median())
cands['189_over_19'] = 189 / 19
cands['total201_over_19'] = 201 / 19
cands['116_over_19+73_over_19 /2'] = (116 / 19 + 73 / 19) / 2
nonzero = cnt[cnt.total > 0]
cands['all_with_any_aha(N18)_mean_total'] = float(nonzero.total.mean())
cands['N18_mean_of_condition_means'] = float((nonzero.control.mean() + nonzero.treatment.mean()) / 2)
res['candidates_5p83'] = cands


def wilcoxon_variants(c, t, label):
    out = {}
    d = np.asarray(c, float) - np.asarray(t, float)
    nz = d[d != 0]
    r = stats.rankdata(np.abs(nz))
    rplus, rminus = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    out['n'] = int(len(d)); out['n_nonzero'] = int(len(nz)); out['r_plus'] = rplus; out['r_minus'] = rminus
    for zm in ['wilcox', 'pratt', 'zsplit']:
        for alt in ['two-sided', 'greater']:
            for meth in ['auto', 'exact', 'asymptotic']:
                for corr in [False, True]:
                    if meth != 'asymptotic' and corr:
                        continue
                    try:
                        w = stats.wilcoxon(c, t, zero_method=zm, alternative=alt, method=meth, correction=corr)
                        out[f'{zm}|{alt}|{meth}|corr={corr}'] = [float(w.statistic), float(w.pvalue)]
                    except Exception as e:
                        out[f'{zm}|{alt}|{meth}|corr={corr}'] = str(e)
    print(f'--- Wilcoxon {label}: n={len(d)}, nonzero={len(nz)}, r+={rplus}, r-={rminus}')
    for k, v in out.items():
        print('   ', k, v)
    return out

res['wilcoxon_counts_all19'] = wilcoxon_variants(cnt.control, cnt.treatment, 'all 19')
res['wilcoxon_counts_N15'] = wilcoxon_variants(n15.control, n15.treatment, 'N15')
n18 = cnt[cnt.total > 0]
res['wilcoxon_counts_N18_excl_P12'] = wilcoxon_variants(n18.control, n18.treatment, 'N18 (excluding P12)')
# paired t for reference
res['paired_t_counts_all19'] = [float(x) for x in stats.ttest_rel(cnt.control, cnt.treatment)]
res['shapiro_diff_all19'] = [float(x) for x in stats.shapiro(cnt.control - cnt.treatment)]

# ---------------- Task 2: time on task ----------------
inv['min_orig'] = inv.dur_orig_s / 60
inv['min_clean'] = inv.dur_clean_s / 60
tt = inv.pivot_table(index='pid', columns='cond', values='min_orig', aggfunc='sum').reindex(PARTICIPANTS)
tc = inv.pivot_table(index='pid', columns='cond', values='min_clean', aggfunc='sum').reindex(PARTICIPANTS)
tt.columns = [f'{c}_min_orig' for c in tt.columns]
tc.columns = [f'{c}_min_clean' for c in tc.columns]
tot = pd.concat([tt, tc], axis=1)
tot['total_min_orig'] = tot.control_min_orig + tot.treatment_min_orig
tot['total_min_clean'] = tot.control_min_clean + tot.treatment_min_clean
tot.to_csv(f'{OUT}/02_time_on_task_per_participant.csv')
print(tot.round(3).to_string())

def summ(df, suffix):
    d = {}
    for cond in ['control', 'treatment']:
        d[f'{cond}_total_min'] = float(df[f'{cond}_min_{suffix}'].sum())
        d[f'{cond}_mean_per_participant'] = float(df[f'{cond}_min_{suffix}'].mean())
        d[f'{cond}_sd_per_participant'] = float(df[f'{cond}_min_{suffix}'].std(ddof=1))
    d['total_min'] = float(df[f'total_min_{suffix}'].sum())
    d['per_participant_total_mean'] = float(df[f'total_min_{suffix}'].mean())
    d['per_participant_total_range'] = [float(df[f'total_min_{suffix}'].min()), float(df[f'total_min_{suffix}'].max())]
    d['per_recording_mean'] = float(d['total_min'] / (4 * len(df)))
    d['n_participants'] = int(len(df))
    return d

res['time_orig_all19'] = summ(tot, 'orig')
res['time_clean_all19'] = summ(tot, 'clean')
tot15 = tot.loc[cnt.index[cnt.in_N15]]
res['time_orig_N15'] = summ(tot15, 'orig')
res['time_clean_N15'] = summ(tot15, 'clean')
tot18 = tot.drop(index='P12')
res['time_orig_N18_excl_P12'] = summ(tot18, 'orig')
res['time_clean_N18_excl_P12'] = summ(tot18, 'clean')

# per-recording stats
res['per_recording_orig_min'] = dict(mean=float(inv.min_orig.mean()), median=float(inv.min_orig.median()),
                                     min=float(inv.min_orig.min()), max=float(inv.min_orig.max()))
res['per_recording_clean_min'] = dict(mean=float(inv.min_clean.mean()), median=float(inv.min_clean.median()),
                                      min=float(inv.min_clean.min()), max=float(inv.min_clean.max()))

# per task (file labels and corrected labels)
pt = []
for lab in ['task', 'task_corrected']:
    for kind in ['min_orig', 'min_clean']:
        g = inv.groupby(lab)[kind].agg(['mean', 'count', 'sum'])
        for task, row in g.iterrows():
            pt.append(dict(label_source=lab, duration=kind, task=task, mean_min=row['mean'],
                           n=int(row['count']), sum_min=row['sum']))
pt = pd.DataFrame(pt)
pt.to_csv(f'{OUT}/02_time_per_task.csv', index=False)
print(pt.pivot_table(index='task', columns=['label_source', 'duration'], values='mean_min').round(3).to_string())
print(pt.pivot_table(index='task', columns=['label_source', 'duration'], values='n').to_string())

# Wilcoxon on minutes
for suffix in ['orig', 'clean']:
    for lab, d in [('all19', tot), ('N15', tot15)]:
        w = stats.wilcoxon(d[f'treatment_min_{suffix}'], d[f'control_min_{suffix}'])
        diff = d[f'treatment_min_{suffix}'] - d[f'control_min_{suffix}']
        res[f'wilcoxon_minutes_{suffix}_{lab}'] = dict(
            W=float(w.statistic), p=float(w.pvalue), n=int(len(d)),
            n_treatment_shorter=int((diff < 0).sum()),
            median_diff_T_minus_C=float(diff.median()), mean_diff_T_minus_C=float(diff.mean()))
        print(f'Wilcoxon minutes {suffix} {lab}:', res[f'wilcoxon_minutes_{suffix}_{lab}'])

json.dump(res, open(f'{OUT}/02_counts_time_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith('wilcoxon_counts')}, indent=1))
