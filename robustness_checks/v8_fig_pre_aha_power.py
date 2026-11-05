"""Figure for the pre-insight power result (paper v08): power in the 2 s before the aha minus power at random
moments of the same recordings (session baseline, both conditions).
(A) frontal region, time x frequency, mean over both conditions, with the cluster from the permutation test
(B) theta (4-8 Hz, -2 to 0 s) per region and condition, mean +- SEM and each participant.
Input: MAPS (maps_v2.npz from v8_pre_aha_power.py). Output: FIG (png and pdf)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from mne.stats import permutation_cluster_1samp_test

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'pdf.fonttype': 42})
z = np.load(os.environ['MAPS'])
FIG = os.environ['FIG']
t, f, ch = z['times'], z['freqs'], list(z['chans'])
pids = sorted({k.split('|')[0] for k in z.files if k.count('|') == 3}, key=lambda s: int(s[1:]))
pids = [p for p in pids if f'{p}|control|aha|session' in z.files and f'{p}|treatment|aha|session' in z.files]
N = len(pids)
ROIS = {'Frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8'], 'Central': ['C3', 'Cz', 'C4'],
        'Parietal': ['P3', 'Pz', 'P4', 'P7', 'P8'], 'Occipital': ['O1', 'O2'], 'Temporal': ['T7', 'T8']}
COL = {'control': '#1f6fd1', 'treatment': '#d62728'}
LAB = {'control': 'Unassisted', 'treatment': 'LLM-assisted'}
tm = (t >= -2.0) & (t <= 0.0)


def diff(p, c, roi):
    ci = [ch.index(x) for x in ROIS[roi]]
    return (z[f'{p}|{c}|aha|session'] - z[f'{p}|{c}|random|session'])[ci].mean(0)


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); r = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p); q[o] = np.clip(np.minimum.accumulate(r[::-1])[::-1], 0, 1); return q


fig = plt.figure(figsize=(10, 3.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.42)

# A: frontal time x frequency, pooled over conditions
X = np.stack([(diff(p, 'control', 'Frontal') + diff(p, 'treatment', 'Frontal'))[:, tm] / 2 for p in pids])
_, cl, pv, _ = permutation_cluster_1samp_test(X, threshold=st.t.ppf(.975, N - 1), n_permutations=2 ** N, tail=0,
                                              out_type='mask', seed=0, verbose='ERROR')
k = int(np.argmin(pv))
ax = fig.add_subplot(gs[0, 0])
M = X.mean(0); v = float(np.ceil(np.abs(M).max() * 2) / 2)
im = ax.pcolormesh(t[tm], f, M, cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest')
ax.contour(t[tm], f, cl[k].astype(float), levels=[0.5], colors='k', linewidths=1.2)
ax.set_xlabel('Time relative to insight (s)'); ax.set_ylabel('Frequency (Hz)')
ax.set_title('Frontal, aha $-$ random moments (both conditions)', fontsize=10.5)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label('Power difference (dB)')
fi, ti = np.where(cl[k])
ax.text(0.02, 0.97, f'cluster {f[fi.min()]:.0f}-{f[fi.max()]:.0f} Hz, {t[tm][ti.min()]:.2f} to {t[tm][ti.max()]:.2f} s, '
        f'p = ' + f'{pv[k]:.3f}'.lstrip('0'), transform=ax.transAxes, va='top', fontsize=8.5,
        bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))

# B: theta per region. POOLED=1 (paper v08): mean of the two conditions, the contrast that was tested, with stars
# from the paper's FDR family (STATS = pre_aha_power_stats.json, 15 region x band tests). POOLED=0: per condition.
POOLED = os.environ.get('POOLED', '1') == '1'
ax = fig.add_subplot(gs[0, 1])
fm = (f >= 4) & (f <= 8)
rows, ps = [], []
for roi in ROIS:
    d = {c: np.array([diff(p, c, roi)[fm][:, tm].mean() for p in pids]) for c in COL}
    rows.append(d); ps.append(st.wilcoxon((d['control'] + d['treatment']) / 2).pvalue)
qs = bh(ps)  # within theta only, used when STATS is not given
if os.environ.get('STATS'):
    import json
    R = {r['roi']: r['q_pooled'] for r in json.load(open(os.environ['STATS']))['v08']['session']['-2..0']
         if r['band'] == 'Theta'}
    qs = [R[roi] for roi in ROIS]
for i, (roi, d) in enumerate(zip(ROIS, rows)):
    if POOLED:
        y = (d['control'] + d['treatment']) / 2
        ax.scatter(np.full(N, i) + np.random.default_rng(i).uniform(-0.08, 0.08, N), y, s=11, color='0.55', alpha=0.5,
                   lw=0)
        m, se = y.mean(), y.std(ddof=1) / np.sqrt(N)
        ax.errorbar(i, m, yerr=se, fmt='o', color='0.15', ms=7, capsize=3.5, lw=1.5)
        continue
    for j, c in enumerate(COL):
        x = i + (j - 0.5) * 0.32
        ax.scatter(np.full(N, x) + np.random.default_rng(i * 2 + j).uniform(-0.05, 0.05, N), d[c], s=9, color=COL[c],
                   alpha=0.35, lw=0)
        m, se = d[c].mean(), d[c].std(ddof=1) / np.sqrt(N)
        ax.errorbar(x, m, yerr=se, fmt='o', color=COL[c], ms=6, capsize=3, lw=1.4, label=LAB[c] if i == 0 else None)
if POOLED:  # stars at one height above all dots: FDR-adjusted p < .05 (*), < .01 (**)
    top = max(((d['control'] + d['treatment']) / 2).max() for d in rows)
    for i in range(len(ROIS)):
        ax.text(i, top + 0.1, '**' if qs[i] < .01 else '*' if qs[i] < .05 else '', ha='center', va='bottom',
                fontsize=12)
    ax.set_ylim(top=top + 0.45)
ax.axhline(0, color='0.3', lw=0.7)
ax.set_xticks(range(len(ROIS))); ax.set_xticklabels(list(ROIS))
ax.set_ylabel('Theta, aha $-$ random (dB)')
ax.set_title('Theta (4-8 Hz), $-2$ to 0 s before the insight', fontsize=10.5)
if not POOLED:
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
for x, lab in zip([fig.axes[0], ax], 'AB'):
    x.text(-0.15, 1.06, lab, transform=x.transAxes, fontsize=15, fontweight='bold')
for ext in ('png', 'pdf'):
    fig.savefig(f'{FIG}.{ext}', dpi=300, bbox_inches='tight')
print('saved', FIG, 'cluster p', float(pv[k]), 'N', N)
