"""Option A analysis 5: Wallas-stage dDTF stats, safe pairing version.

Reads the already-computed dDTF windows at
    optB_ddtf/wallas_mean/windows.csv.gz   (averaging over channels within each ROI)
    optB_ddtf/wallas_pca/windows.csv.gz    (first-PC per ROI, Tak's robustness check)

For each (band, pathway):
  - keep only participants who have BOTH treatment and control data in the dataset
  - per-participant mean dDTF across the stages they contributed to, within each condition
  - paired Wilcoxon treatment vs control, FDR across the 12 tests
  - also fit the paper's LRT mixed model with safe reindex + dropna so missing
    (participant, stage, condition) cells drop out instead of crashing

Writes: optB_5_ddtf_wallas.json (both wallas_mean and wallas_pca).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import RESULTS, ERP_DIR, WHOLE_SESSION_DIR  # noqa: E402
import os, sys  # noqa: E402
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, WHOLE_SESSION_DIR)
sys.path.insert(0, ERP_DIR)
import robust_stats as rs

STAGES = ['Preparation', 'Idea generation', 'Implementation', 'Verification']
OPTA = os.path.join(RESULTS, 'optB_ddtf')
OUT = RESULTS


def _fit(formula, d, re):
    import statsmodels.formula.api as smf
    last = None
    for method in ['lbfgs', 'bfgs', 'powell', 'nm']:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return smf.mixedlm(formula, d, groups=d['pid'], re_formula=re).fit(reml=False, method=method)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def lrt_safe(df, feat):
    """LRT of condition on cell_means, with safe handling of missing cells.
    df = per-(pid, cond, seg, feat) cell means. We dropna on feat first.
    """
    d = df[['pid', 'cond', 'seg', feat]].dropna().copy()
    d = d.rename(columns={feat: 'y'})
    if d.empty:
        return np.nan, np.nan, np.nan
    d['y'] = d['y'] / d['y'].std()
    try:
        m1 = _fit('y ~ C(cond) * C(seg)', d, '~1')
        m0 = _fit('y ~ C(seg)', d, '~1')
        lam = 2 * (m1.llf - m0.llf)
        dfree = len(m1.fe_params) - len(m0.fe_params)
        return float(lam), int(dfree), float(st.chi2.sf(max(lam, 0), dfree))
    except Exception:
        return np.nan, np.nan, np.nan


def analyze(tag):
    path = os.path.join(OPTA, tag, 'windows.csv.gz')
    if not os.path.exists(path):
        return dict(error='missing windows.csv.gz', path=path)
    win = pd.read_csv(path)
    stable = win[win.stable].copy()
    feats = [c for c in stable.columns if '->' in c]
    # keep only stages used in the paper
    stable = stable[stable.seg.isin(STAGES)].copy()

    # Per-(pid, cond, seg) cell means  (unchanged from paper, drives the LRT)
    cell = stable.groupby(['pid', 'cond', 'seg'])[feats].mean().reset_index()

    # paired within-subject means over the stages each person contributed to
    pm = stable.groupby(['pid', 'cond'])[feats].mean().unstack('cond')
    # keep only pids who have BOTH conditions
    rows = []
    N_total = None
    for f in feats:
        try:
            c = pm[(f, 'control')]
            t = pm[(f, 'treatment')]
        except KeyError:
            rows.append(dict(pathway=f, error='condition missing'))
            continue
        mask = c.notna() & t.notna()
        c, t = c[mask], t[mask]
        n = len(c)
        N_total = N_total or n
        if n < 3:
            rows.append(dict(pathway=f, N=n, error='<3 paired participants'))
            continue
        try:
            w = rs.exact_wilcoxon(c.values, t.values)
            p_paired = w['p']
        except Exception as e:  # noqa: BLE001
            p_paired = float(st.wilcoxon(c.values, t.values).pvalue)
        chi2, df_, p_lrt = lrt_safe(cell, f)
        rows.append(dict(pathway=f,
                         mean_unassisted=float(c.mean()),
                         mean_llm=float(t.mean()),
                         change_pct=float(100 * (t.mean() - c.mean()) / c.mean()),
                         n_lower_with_llm=int((t.values < c.values).sum()),
                         N=n,
                         chi2_paper=chi2, df_paper=df_, p_paper=p_lrt,
                         p_paired=p_paired))

    df_rows = pd.DataFrame(rows)
    # FDR across the pathways where we have a p
    for col in ('p_paired', 'p_paper'):
        valid = df_rows[col].notna() if col in df_rows else pd.Series([False] * len(df_rows))
        q = np.full(len(df_rows), np.nan)
        if valid.any():
            q[valid.values] = rs.fdr_bh(df_rows.loc[valid, col].values)
        df_rows['q_' + col.split('_', 1)[1]] = q
    return dict(tag=tag, N=N_total, stats=df_rows.round(5).to_dict(orient='records'),
                pids_used=list(sorted(pm.dropna(how='any', subset=[(feats[0], 'control'), (feats[0], 'treatment')]).index)) if feats else [])


out = {}
for tag in ('wallas_mean', 'wallas_pca'):
    print(f'=== {tag} ===', flush=True)
    out[tag] = analyze(tag)
    stats_df = pd.DataFrame(out[tag].get('stats', []))
    if not stats_df.empty:
        print(stats_df.sort_values('q_paper').to_string(index=False))

with open(os.path.join(OUT, 'optB_5_ddtf_wallas.json'), 'w') as fh:
    json.dump(dict(stats=out), fh, indent=1)
print('\nwrote optB_5_ddtf_wallas.json')
