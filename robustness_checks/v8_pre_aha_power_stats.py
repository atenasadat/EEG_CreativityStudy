"""Stats for pre_aha_v2: aha - random (insight-specific power) and its condition difference.
Families (Wilcoxon, BH-FDR within each): 5 ROIs x theta/alpha/beta in -2..0 and -1..0 s.
Literature-driven tests (Jung-Beeman et al. 2004, Kounios & Beeman 2014): right posterior alpha (P4, P8, O2) in -2..-0.5 s,
right temporal gamma (T8, 31-49 Hz) in -0.5..0 s (speech starts at 0, so gamma may carry muscle activity).
Also a time x frequency cluster test (-2..0 s, 1-49 Hz) per ROI for the condition x type contrast."""
import sys, json, numpy as np, scipy.stats as st
from mne.stats import permutation_cluster_1samp_test
import os
MAPS = os.environ['MAPS']
ROIS = {'Frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8'], 'Central': ['C3', 'Cz', 'C4'], 'Parietal': ['P3', 'Pz', 'P4', 'P7', 'P8'],
        'Occipital': ['O1', 'O2'], 'Temporal': ['T7', 'T8'], 'RightPost': ['P4', 'P8', 'O2'], 'RightTemp': ['T8'], 'F3C3P3': ['F3', 'C3', 'P3']}
BANDS = {'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (14, 30), 'Gamma': (31, 49)}
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p); q[o] = np.clip(np.minimum.accumulate(r[::-1])[::-1], 0, 1); return q
res = {}
for ver in ['v08']:
    z = np.load(MAPS); t, f, ch = z['times'], z['freqs'], list(z['chans'])
    pids = sorted({k.split('|')[0] for k in z.files if k.count('|') == 3}, key=lambda s: int(s[1:]))
    pids = [p for p in pids if f'{p}|control|aha|session' in z.files and f'{p}|treatment|aha|session' in z.files]
    print(f'\n######## {ver} files, N = {len(pids)}')
    res[ver] = {}
    for base in ['session', 'early']:
        def val(p, c, ty, roi, band, t0, t1):
            m = z[f'{p}|{c}|{ty}|{base}']; ci = [ch.index(x) for x in ROIS[roi]]
            fm = (f >= BANDS[band][0]) & (f <= BANDS[band][1]); tm = (t >= t0) & (t <= t1)
            return m[ci][:, fm][:, :, tm].mean()
        def contrast(roi, band, t0, t1):
            d = {c: np.array([val(p, c, 'aha', roi, band, t0, t1) - val(p, c, 'random', roi, band, t0, t1) for p in pids]) for c in ['control', 'treatment']}
            pooled = (d['control'] + d['treatment']) / 2; cx = d['treatment'] - d['control']
            return dict(roi=roi, band=band, window=f'{t0:g}..{t1:g}', aha_minus_random_U=float(d['control'].mean()), aha_minus_random_LLM=float(d['treatment'].mean()),
                        p_pooled=float(st.wilcoxon(pooled).pvalue), p_U=float(st.wilcoxon(d['control']).pvalue), p_LLM=float(st.wilcoxon(d['treatment']).pvalue),
                        cond_x_type=float(cx.mean()), n_LLM_more=int((cx > 0).sum()), p_cond_x_type=float(st.wilcoxon(cx).pvalue))
        print(f'--- baseline {base}')
        out = {}
        for (t0, t1) in [(-2.0, 0.0), (-1.0, 0.0)]:
            rows = [contrast(r, b, t0, t1) for r in ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal'] for b in ['Theta', 'Alpha', 'Beta']]
            for key in ['p_pooled', 'p_cond_x_type']:
                q = bh([r[key] for r in rows])
                for r, qq in zip(rows, q): r['q' + key[1:]] = float(qq)
            out[f'{t0:g}..{t1:g}'] = rows
            print(f'  window {t0:g}..{t1:g}: aha vs random (both conditions) FDR<.05: {sum(r["q_pooled"] < .05 for r in rows)} | condition x type FDR<.05: {sum(r["q_cond_x_type"] < .05 for r in rows)}')
            for r in rows:
                if r['p_pooled'] < .05 or r['p_cond_x_type'] < .05:
                    print(f'     {r["roi"]:9s} {r["band"]:5s} aha-rand U {r["aha_minus_random_U"]:+.2f} LLM {r["aha_minus_random_LLM"]:+.2f} dB  p_pooled {r["p_pooled"]:.3f} q {r["q_pooled"]:.3f} | LLM-U {r["cond_x_type"]:+.2f} dB ({r["n_LLM_more"]}/{len(pids)}) p {r["p_cond_x_type"]:.3f} q {r["q_cond_x_type"]:.3f}')
        lit = [contrast('RightPost', 'Alpha', -2.0, -0.5), contrast('RightTemp', 'Gamma', -0.5, 0.0), contrast('F3C3P3', 'Theta', -0.45, -0.30),
               contrast('F3C3P3', 'Alpha', -0.45, -0.30), contrast('F3C3P3', 'Beta', -0.45, -0.30)]
        out['literature'] = lit
        print('  literature / ERP-linked tests:')
        for r in lit:
            print(f'     {r["roi"]:9s} {r["band"]:5s} {r["window"]:11s} aha-rand U {r["aha_minus_random_U"]:+.2f} (p {r["p_U"]:.3f}) LLM {r["aha_minus_random_LLM"]:+.2f} (p {r["p_LLM"]:.3f}) pooled p {r["p_pooled"]:.3f} | LLM-U {r["cond_x_type"]:+.2f} p {r["p_cond_x_type"]:.3f}')
        if base == 'session':
            tm = (t >= -2.0) & (t <= 0.0); thr = st.t.ppf(.975, len(pids) - 1); cl_out = {}
            for roi in ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal']:
                ci = [ch.index(x) for x in ROIS[roi]]
                Xc = np.stack([(z[f'{p}|treatment|aha|session'] - z[f'{p}|treatment|random|session'] - z[f'{p}|control|aha|session'] + z[f'{p}|control|random|session'])[ci].mean(0)[:, tm] for p in pids])
                Xp = np.stack([(z[f'{p}|treatment|aha|session'] - z[f'{p}|treatment|random|session'] + z[f'{p}|control|aha|session'] - z[f'{p}|control|random|session'])[ci].mean(0)[:, tm] / 2 for p in pids])
                for nm, XX in [('cond_x_type', Xc), ('aha_vs_random', Xp)]:
                    _, cl, pv, _ = permutation_cluster_1samp_test(XX, threshold=thr, n_permutations=2 ** len(pids), tail=0, out_type='mask', seed=0, verbose='ERROR')
                    best = None
                    for m, p in zip(cl, pv):
                        fi, ti = np.where(m)
                        r = dict(p=float(p), f=f'{f[fi.min()]:.0f}-{f[fi.max()]:.0f} Hz', t=f'{t[tm][ti.min()]:.2f}..{t[tm][ti.max()]:.2f} s', sign='+' if XX[:, m].mean() > 0 else '-')
                        if best is None or p < best['p']: best = r
                    cl_out[f'{roi} {nm}'] = best
                    print(f'  TF cluster {roi:9s} {nm:13s}: {best}')
            out['tf_cluster'] = cl_out
        res[ver][base] = out
json.dump(res, open(os.path.join(os.path.dirname(MAPS), 'pre_aha_power_stats.json'), 'w'), indent=1)
print('DONE')
