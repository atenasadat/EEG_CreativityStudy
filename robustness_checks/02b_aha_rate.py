"""Aha rate (ahas per 10 min of original-clock time on task) per participant x condition.

Input : outputs/02_aha_counts_per_participant.csv, outputs/02_time_on_task_per_participant.csv
Output: outputs/02b_aha_rate_per_participant.csv, outputs/02b_aha_rate_summary.json
Wilcoxon convention as the count test: scipy.stats.wilcoxon(control, treatment, zero_method='wilcox',
two-sided; method 'auto').
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from common import *

cnt = pd.read_csv(f'{OUT}/02_aha_counts_per_participant.csv', index_col=0)
tim = pd.read_csv(f'{OUT}/02_time_on_task_per_participant.csv', index_col=0)
df = pd.DataFrame({'aha_control': cnt.control, 'aha_treatment': cnt.treatment,
                   'min_control': tim.control_min_orig, 'min_treatment': tim.treatment_min_orig,
                   'in_N15': cnt.in_N15})
df['rate10_control'] = df.aha_control / df.min_control * 10
df['rate10_treatment'] = df.aha_treatment / df.min_treatment * 10
df.to_csv(f'{OUT}/02b_aha_rate_per_participant.csv')
print(df.round(4).to_string())

res = {}
for lab, d in [('all19', df), ('N15', df[df.in_N15.astype(bool)])]:
    u, l = d.rate10_control.values, d.rate10_treatment.values
    diff = u - l
    w = stats.wilcoxon(u, l, zero_method='wilcox', alternative='two-sided')
    nz = diff[diff != 0]; rk = stats.rankdata(np.abs(nz))
    res[lab] = dict(
        N=int(len(d)),
        unassisted_mean=float(u.mean()), unassisted_sd=float(u.std(ddof=1)), unassisted_median=float(np.median(u)),
        llm_mean=float(l.mean()), llm_sd=float(l.std(ddof=1)), llm_median=float(np.median(l)),
        totals_rate10_unassisted=float(d.aha_control.sum() / d.min_control.sum() * 10),
        totals_rate10_llm=float(d.aha_treatment.sum() / d.min_treatment.sum() * 10),
        totals=dict(aha_u=int(d.aha_control.sum()), min_u=float(d.min_control.sum()),
                    aha_llm=int(d.aha_treatment.sum()), min_llm=float(d.min_treatment.sum())),
        W=float(w.statistic), p=float(w.pvalue), n_nonzero=int(len(nz)),
        r_plus=float(rk[nz > 0].sum()), r_minus=float(rk[nz < 0].sum()),
        n_lower_with_llm=int((l < u).sum()), n_higher_with_llm=int((l > u).sum()), n_equal=int((l == u).sum()),
        median_diff_U_minus_LLM=float(np.median(diff)))
json.dump(res, open(f'{OUT}/02b_aha_rate_summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
