"""Follow-up figures for Richard (Oct 28): the pre-aha ERP at F3, C3 and P3 plotted
separately, and each person's window mean in both conditions.
Same data as the lab slide: outputs/03_participant_erps.npz, N = 15."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import RESULTS, OUT  # noqa: E402
import os, sys  # noqa: E402
import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

z = np.load(os.path.join(OUT, '03_participant_erps.npz'))
times = z['times']
chans = list(z['chans'])
U_all = z['control'] * 1e6
L_all = z['treatment'] * 1e6
N = U_all.shape[0]
win = (times >= -0.4501) & (times <= -0.2966)
thr = st.t.ppf(0.975, N - 1)
OUT = RESULTS + '/'
COL = {'Without LLM': '#1f6fd1', 'With LLM': '#d62728'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=200, sharey=True)
for ax, ch in zip(axes, ['F3', 'C3', 'P3']):
    i = chans.index(ch)
    U, L = U_all[:, i], L_all[:, i]
    ax.axvspan(times[win][0], times[win][-1], color='#e6e6e6', lw=0, zorder=0)
    for Y, lab in [(U, 'Without LLM'), (L, 'With LLM')]:
        m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(N)
        ax.fill_between(times, m - se, m + se, color=COL[lab], alpha=0.16, lw=0)
        ax.plot(times, m, color=COL[lab], lw=2.0, label=lab)
    # samples where this channel alone passes the cluster-forming threshold
    t = st.ttest_rel(U, L, axis=0).statistic
    above = np.abs(t) > thr
    ax.plot(np.where(above, times, np.nan), np.full_like(times, -9.3), color='#111111', lw=4, solid_capstyle='butt')
    ax.axhline(0, color='#bbbbbb', lw=0.8)
    ax.axvline(0, color='#111111', ls='--', lw=1.0)
    ax.set_xlim(-0.5, 0.3)
    ax.set_ylim(-10, 10)
    ax.set_title(ch, fontsize=15, fontweight='bold')
    ax.set_xlabel('Time from aha (s)')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    du, dl = U[:, win].mean(1), L[:, win].mean(1)
    print(f'{ch}: without {du.mean():+.2f}, with {dl.mean():+.2f} uV, '
          f'LLM higher in {int((dl > du).sum())}/{N}, Wilcoxon p = {st.wilcoxon(du, dl).pvalue:.4f}, '
          f'above-threshold span {times[above & (times < 0)].min():.3f}..{times[above & (times < 0)].max():.3f} s')
axes[0].set_ylabel('µV')
axes[2].legend(frameon=False, loc='upper right', fontsize=12)
fig.tight_layout()
fig.savefig(OUT + 'erp_F3_C3_P3_separate.png', facecolor='white')

# each person's window mean (average of F3, C3, P3), both conditions
roi = [chans.index(c) for c in ['F3', 'C3', 'P3']]
du = U_all[:, roi][:, :, win].mean(axis=(1, 2))
dl = L_all[:, roi][:, :, win].mean(axis=(1, 2))
fig, ax = plt.subplots(figsize=(4.2, 4.4), dpi=200)
for a, b in zip(du, dl):
    ax.plot([0, 1], [a, b], color='#9a9a9a' if b > a else '#111111', lw=1.2, zorder=1)
ax.scatter(np.zeros(N), du, color=COL['Without LLM'], s=36, zorder=2)
ax.scatter(np.ones(N), dl, color=COL['With LLM'], s=36, zorder=2)
ax.axhline(0, color='#bbbbbb', lw=0.8)
ax.set_xticks([0, 1], ['Without LLM', 'With LLM'])
ax.set_xlim(-0.35, 1.35)
ax.set_ylabel('F3, C3, P3, -0.45 to -0.30 s (µV)')
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT + 'erp_window_per_person.png', facecolor='white')
print(f'per person: LLM higher in {int((dl > du).sum())}/{N}, Wilcoxon p = {st.wilcoxon(du, dl).pvalue:.5f}, '
      f'range of differences {np.min(dl - du):+.1f} to {np.max(dl - du):+.1f} uV')
