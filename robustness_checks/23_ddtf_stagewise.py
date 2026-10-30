"""Stage-wise directed connectivity (dDTF) as in the paper's Methods, with channel-average or PCA region signals.

Rewrite of Athena's wallas_analysis/wallas_mapping.ipynb dDTF part (not on GitHub). Uses her unchanged functions
from Whole Session/wholesession_ddtf.py (commit 8c85cba) for loading, rest-offset correction, 4-s windows that do
not span removed data, resampling to 128 Hz, detrend/z-score, MVAR least squares, stability check and dDTF.

Region signal (--roi):
  mean : average of the region's channels (paper)
  pca  : first principal component of the region's channels, fitted once per participant on all task data of
         BOTH conditions (1-45 Hz band-passed copy for the fit), unit-norm weights; same filter for both conditions.
Segment label (--segments):
  wallas  : Wallas stage of the coding segment that contains the window centre (needs creativitycodings.csv,
            --coding PATH). Video clock -> EEG clock offset per session = median(EEG aha time on the original,
            uncut clock - coding-sheet aha time) from aha_events.json; windows are mapped back through the removed
            segments. Illumination is not used (as in the paper).
  quarter : session quarter (1-4) by the window's position in the cleaned recording. Stand-in for the four coded
            stages when the coding sheet is not available.
Statistics per pathway (12 = 6 directed pathways x theta/alpha), on participant x condition x segment means:
  paper model : LME, participant random intercept, ML; dDTF ~ condition * segment vs dDTF ~ segment; LRT chi2(4)
  strict model: same with a participant random slope for condition in both models
  paired      : participant means over segments, exact paired Wilcoxon
  BH-FDR across the 12 pathways for each.
Usage: python 23_ddtf_stagewise.py --roi mean|pca --segments quarter|wallas [--coding CSV] [--order 5] --out DIR
       python 23_ddtf_stagewise.py --check-clock --out DIR     (tests the clock mapping on aha_events.json)
       python 23_ddtf_stagewise.py --aic --out DIR             (model order by AIC, 1000 random windows)
"""
import argparse
import json
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import signal
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import WHOLE_SESSION_DIR, AHA_JSON  # noqa: E402

TOOLS = None
sys.path.insert(0, WHOLE_SESSION_DIR)
import wholesession_ddtf as W  # noqa: E402
import robust_stats as rs      # noqa: E402  (her ERP/robust_stats.py, put on the path by wholesession_ddtf)

AHA_JSON = AHA_JSON
STAGES = ['Preparation', 'Idea generation', 'Implementation', 'Verification']
# Readme section 11: file task label -> coding-sheet task label
TASK_FIX = {('P12', 'control', 'B1'): 'B2', ('P23', 'control', 'B3'): 'B2', ('P17', 'control', 'A2'): 'B2'}
FS_NEW = 128.0


def parse_file(fname):
    m = re.match(r'(P\d+)_(control|treatment)_([AB]\d)_postcleaning\d+\.set$', os.path.basename(fname))
    return m.groups()


def hms(s):
    h, m, x = str(s).strip().split(':')
    return int(h) * 3600 + int(m) * 60 + float(x)


def removed_before(raw):
    """(cleaned onset, cumulative removed seconds up to and including this boundary) for every boundary."""
    a = raw.annotations
    rows = [(float(o), float(d)) for o, d, de in zip(a.onset, a.duration, a.description) if str(de) == 'boundary']
    rows.sort()
    cum, out = 0.0, []
    for o, d in rows:
        cum += d
        out.append((o, cum))
    return out


def cleaned_to_original(t, rb):
    """Original (uncut) clock for a cleaned-clock time."""
    add = 0.0
    for o, cum in rb:
        if o <= t + 1e-9:
            add = cum
        else:
            break
    return t + add


def session_offset(raw, pid, cond, task, aha):
    """Median(EEG original-clock aha time - coding-sheet aha time) and the residuals."""
    key = f"{cond} - {TASK_FIX.get((pid, cond, task), task)}"
    if pid not in aha or key not in aha[pid]:
        return None, []
    sheet = sorted(hms(x) for x in aha[pid][key]['times'])
    a = raw.annotations
    rb = removed_before(raw)
    eeg = sorted(cleaned_to_original(float(o), rb) for o, de in zip(a.onset, a.description) if str(de) == '100')
    if not eeg or not sheet:
        return None, []
    # pair each EEG aha with the sheet time that gives the most consistent offset
    cands = np.array([e - s for e in eeg for s in sheet])
    best = None
    for c in cands:
        res = [min(abs(e - s - c) for s in sheet) for e in eeg]
        score = np.median(res)
        if best is None or score < best[0]:
            best = (score, c)
    off = best[1]
    pairs = [(e, min(sheet, key=lambda s: abs(e - s - off))) for e in eeg]
    off = float(np.median([e - s for e, s in pairs]))
    resid = [float(e - s - off) for e, s in pairs]
    return off, resid


def pca_weights(datas, ch_names):
    data = np.concatenate(datas, axis=1)
    out = {}
    for roi, chans in W.ROIS.items():
        idx = [ch_names.index(c) for c in chans if c in ch_names]
        x = data[idx] - data[idx].mean(axis=1, keepdims=True)
        evals, evecs = np.linalg.eigh(np.cov(x))
        w = evecs[:, -1]
        w = w if w.sum() >= 0 else -w
        out[roi] = (idx, w / np.linalg.norm(w), float(100 * evals[-1] / evals.sum()))
    return out


def roi_signals(raw, ch_names, mode, weights):
    if mode == 'mean':
        return W.roi_series(raw, ch_names)
    data = raw.copy().pick(ch_names).get_data()
    rows, names = [], []
    for roi, (idx, w, _) in weights.items():
        rows.append(w @ data[idx])
        names.append(roi)
    return np.asarray(rows), names


def load_coding(path):
    c = pd.read_csv(path)
    c = c[c['Wallas Step'].notna()].copy()
    c['t0'] = c['Timestamp'].map(hms)
    c['cond'] = c['condition'].map({'C': 'control', 'T': 'treatment'})
    c = c.sort_values(['Pid', 'name', 'cond', 't0'])
    out = {}
    for (pid, task, cond), g in c.groupby(['Pid', 'name', 'cond']):
        t0 = g.t0.values
        t1 = np.r_[t0[1:], t0[-1] + g.duration_seconds.values[-1] if 'duration_seconds' in g else t0[-1] + 60]
        out[(pid, cond, task)] = list(zip(t0, t1, g['Wallas Step'].str.strip().values))
    return out


def run(args):
    os.makedirs(args.out, exist_ok=True)
    aha = json.load(open(AHA_JSON))
    coding = load_coding(args.coding) if args.segments == 'wallas' else None
    pids, c_groups, t_groups = W.get_condition_file_groups(W.BASE_PATH)
    freqs = np.arange(1.0, 50.0 + 0.5, 0.5)
    bands = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in W.DDTF_BANDS.items()}
    rows, wrows, clock_rows = [], {}, []
    for pid in pids:
        first = W.read_eeg_file(c_groups[pid][0], base_path=W.BASE_PATH, preload=True, channels=W.STANDARD_CHANNELS)
        ch_names = W.available_channels(first, W.STANDARD_CHANNELS)
        baseline_raw, _ = W.load_baseline_raw(pid, first, ch_names)
        raws = []
        for cond, files in [('control', c_groups[pid]), ('treatment', t_groups[pid])]:
            for f in files:
                raw = W.read_eeg_file(f, base_path=W.BASE_PATH, preload=True, channels=ch_names)
                raw = W.baseline_correct_raw(raw, baseline_raw, ch_names)
                raws.append((cond, f, raw))
        weights = None
        if args.roi == 'pca':
            weights = pca_weights([r.copy().filter(1.0, 45.0, verbose='ERROR').pick(ch_names).get_data()
                                   for _, _, r in raws], ch_names)
            wrows[pid] = {roi: dict(pc1_var_pct=v, weights={ch_names[i]: float(x) for i, x in zip(idx, w)})
                          for roi, (idx, w, v) in weights.items()}
        for cond, f, raw in raws:
            _, _, task = parse_file(f)
            sf = float(raw.info['sfreq'])
            data, names = roi_signals(raw, ch_names, args.roi, weights)
            starts, n = W.window_starts(raw, 4.0)
            dur = raw.n_times / sf
            if coding is not None:
                off, resid = session_offset(raw, pid, cond, task, aha)
                rb = removed_before(raw)
                segs = coding.get((pid, cond, TASK_FIX.get((pid, cond, task), task)))
                clock_rows.append(dict(pid=pid, cond=cond, task=task, offset_s=off,
                                       max_abs_resid_s=max(map(abs, resid)) if resid else np.nan,
                                       has_coding=segs is not None))
            for s0 in starts:
                win = data[:, s0:s0 + n]
                if win.shape[1] != n:
                    continue
                centre = (s0 + n / 2) / sf
                if args.segments == 'quarter':
                    seg = f'Q{min(int(4 * centre / dur), 3) + 1}'
                else:
                    if off is None or segs is None:
                        continue
                    video = cleaned_to_original(centre, rb) - off
                    seg = next((lab for a, b, lab in segs if a <= video < b), None)
                    if seg not in STAGES:
                        continue
                x = signal.resample_poly(win, int(FS_NEW), int(round(sf)), axis=1)
                vals, stable = W.ddtf_window(x, FS_NEW, args.order, freqs, bands)
                if vals is None or not stable:
                    rows.append(dict(pid=pid, cond=cond, task=task, seg=seg, stable=False))
                    continue
                r = dict(pid=pid, cond=cond, task=task, seg=seg, stable=True)
                for b in W.DDTF_BANDS:
                    for si, src in enumerate(names):
                        for ti, tgt in enumerate(names):
                            if si != ti:
                                r[f'{b} {src}->{tgt}'] = float(vals[b][ti, si])
                rows.append(r)
        print(pid, 'windows so far', len(rows), flush=True)
    win_df = pd.DataFrame(rows)
    win_df.to_csv(os.path.join(args.out, 'windows.csv.gz'), index=False)
    if wrows:
        json.dump(wrows, open(os.path.join(args.out, 'pca_weights.json'), 'w'), indent=1)
    if clock_rows:
        pd.DataFrame(clock_rows).to_csv(os.path.join(args.out, 'clock_offsets.csv'), index=False)
    stats(win_df, args.out)


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


def lrt(df, feat, slope):
    """LRT of condition (main + x segment). y is scaled by its SD: the LRT does not change, the fit is stabler."""
    d = df.rename(columns={feat: 'y'}).copy()
    d['y'] = d['y'] / d['y'].std()
    re = '~C(cond)' if slope else '~1'
    m1 = _fit('y ~ C(cond) * C(seg)', d, re)
    m0 = _fit('y ~ C(seg)', d, re)
    lam = 2 * (m1.llf - m0.llf)
    dfree = len(m1.fe_params) - len(m0.fe_params)
    return float(lam), int(dfree), float(st.chi2.sf(max(lam, 0), dfree))


def stats(win_df, out):
    ok = win_df[win_df.stable]
    feats = [c for c in ok.columns if '->' in c]
    cell = ok.groupby(['pid', 'cond', 'seg'])[feats].mean().reset_index()
    cell.to_csv(os.path.join(out, 'cell_means.csv'), index=False)
    pm = ok.groupby(['pid', 'cond'])[feats].mean()
    rows = []
    for f in feats:
        c = pm.xs('control', level='cond')[f]
        t = pm.xs('treatment', level='cond')[f].loc[c.index]
        l1, d1, p1 = lrt(cell, f, slope=False)
        try:
            l2, d2, p2 = lrt(cell, f, slope=True)
        except Exception as e:  # noqa: BLE001
            l2, d2, p2 = np.nan, np.nan, np.nan
            print('strict model failed', f, e)
        w = rs.exact_wilcoxon(c.values, t.values)
        mu_c = cell[cell.cond == 'control'][f].mean()
        mu_t = cell[cell.cond == 'treatment'][f].mean()
        rows.append(dict(pathway=f, mean_unassisted=mu_c, mean_llm=mu_t, change_pct=100 * (mu_t - mu_c) / mu_c,
                         n_lower_with_llm=int((t < c).sum()), N=len(c),
                         chi2_paper=l1, df_paper=d1, p_paper=p1, chi2_strict=l2, df_strict=d2, p_strict=p2,
                         p_paired=w['p']))
    s = pd.DataFrame(rows)
    for col in ['p_paper', 'p_strict', 'p_paired']:
        s[col.replace('p_', 'q_')] = rs.fdr_bh(s[col].fillna(1).values)
    s.to_csv(os.path.join(out, 'stats.csv'), index=False)
    n_win = dict(total=len(win_df), stable=int(win_df.stable.sum()))
    json.dump(n_win, open(os.path.join(out, 'n_windows.json'), 'w'))
    pd.set_option('display.width', 250)
    print(n_win)
    print(s[['pathway', 'mean_unassisted', 'mean_llm', 'change_pct', 'n_lower_with_llm', 'chi2_paper', 'q_paper',
             'chi2_strict', 'q_strict', 'q_paired']].sort_values('q_paper').round(4).to_string(index=False))


def check_clock(args):
    """Clock mapping test: every EEG aha must land on a coding-sheet aha after one offset per session."""
    os.makedirs(args.out, exist_ok=True)
    aha = json.load(open(AHA_JSON))
    pids, c_groups, t_groups = W.get_condition_file_groups(W.BASE_PATH)
    rows = []
    for pid in pids:
        for f in c_groups[pid] + t_groups[pid]:
            _, cond, task = parse_file(f)
            raw = W.read_eeg_file(f, base_path=W.BASE_PATH, preload=False)
            off, resid = session_offset(raw, pid, cond, task, aha)
            rows.append(dict(file=os.path.basename(f), pid=pid, cond=cond, task=task, n_eeg_aha=len(resid),
                             offset_s=off, max_abs_resid_s=max(map(abs, resid)) if resid else np.nan,
                             median_abs_resid_s=float(np.median(np.abs(resid))) if resid else np.nan))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(args.out, 'clock_check.csv'), index=False)
    with_aha = t[t.n_eeg_aha > 0]
    print(f"sessions with EEG ahas: {len(with_aha)}; mapped: {with_aha.offset_s.notna().sum()}; "
          f"median |resid| {with_aha.median_abs_resid_s.median():.2f} s; max |resid| {with_aha.max_abs_resid_s.max():.2f} s")


def aic(args):
    """AIC(p) = ln|Sigma_p| + 2 p k^2 / (N - p), averaged over 1000 random windows (paper formula)."""
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    pids, c_groups, t_groups = W.get_condition_file_groups(W.BASE_PATH)
    wins = []
    for pid in pids:
        for f in c_groups[pid] + t_groups[pid]:
            raw = W.read_eeg_file(f, base_path=W.BASE_PATH, preload=True, channels=W.STANDARD_CHANNELS)
            ch = W.available_channels(raw, W.STANDARD_CHANNELS)
            data, _ = W.roi_series(raw, ch)
            starts, n = W.window_starts(raw, 4.0)
            for s0 in starts:
                wins.append((data, s0, n, float(raw.info['sfreq'])))
    pick = rng.choice(len(wins), size=min(1000, len(wins)), replace=False)
    orders = range(1, 21)
    vals = np.zeros((len(pick), len(orders)))
    for k, i in enumerate(pick):
        data, s0, n, sf = wins[i]
        x = signal.resample_poly(data[:, s0:s0 + n], int(FS_NEW), int(round(sf)), axis=1)
        x = signal.detrend(x, axis=1)
        x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        for j, p in enumerate(orders):
            a, sigma, _ = W.fit_mvar(x, p)
            N = x.shape[1]
            vals[k, j] = np.log(np.linalg.det(sigma)) + 2 * p * 9 / (N - p)
    m = vals.mean(0)
    best = int(list(orders)[int(np.argmin(m))])
    json.dump(dict(aic_mean=m.tolist(), best_order=best, n_windows=len(pick)), open(os.path.join(args.out, 'aic.json'), 'w'))
    print('AIC best order', best, np.round(m, 4).tolist())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--roi', choices=['mean', 'pca'], default='mean')
    ap.add_argument('--segments', choices=['quarter', 'wallas'], default='quarter')
    ap.add_argument('--coding', default=None)
    ap.add_argument('--order', type=int, default=5)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-clock', action='store_true')
    ap.add_argument('--aic', action='store_true')
    ap.add_argument('--stats-only', action='store_true', help='recompute statistics from <out>/windows.csv.gz')
    a = ap.parse_args()
    if a.stats_only:
        stats(pd.read_csv(os.path.join(a.out, 'windows.csv.gz')), a.out)
    elif a.check_clock:
        check_clock(a)
    elif a.aic:
        aic(a)
    else:
        if a.segments == 'wallas' and not a.coding:
            sys.exit('--segments wallas needs --coding creativitycodings.csv')
        run(a)
