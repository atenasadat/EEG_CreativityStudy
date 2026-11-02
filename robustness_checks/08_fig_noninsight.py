"""Task 7 figure: non-insight control.
Input : outputs/03_participant_erps.npz, outputs/07_noninsight_erps.npz, outputs/07_noninsight_results.json
Output: Figures/v04/fig_noninsight_control.{png,pdf}
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from common import *

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 10, 'pdf.fonttype': 42})
a = np.load(f'{OUT}/03_participant_erps.npz'); r = np.load(f'{OUT}/07_noninsight_erps.npz')
S = json.load(open(f'{OUT}/07_noninsight_results.json'))
times = a['times']; chans = list(a['chans'])
roi = [chans.index(c) for c in ['F3', 'C3', 'P3']]
win = (times >= -0.4501) & (times <= -0.2966)
R = lambda X: X[:, roi].mean(1) * 1e6  # (N, t) ROI waveform per participant
AC, AT, RC, RT = R(a['control']), R(a['treatment']), R(r['control']), R(r['treatment'])
N = AC.shape[0]

def band(ax, Y, col, ls, lab):
    m = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(N)
    ax.fill_between(times, m - se, m + se, color=col, alpha=0.18, lw=0)
    ax.plot(times, m, color=col, ls=ls, lw=1.5, label=lab)

def deco(ax, title):
    ax.axvspan(times[win][0], times[win][-1], color='0.87', lw=0, zorder=0)
    ax.axhline(0, color='0.3', lw=0.6); ax.axvline(0, color='k', ls='--', lw=0.9)
    ax.set_xlim(-0.5, 0.5); ax.set_title(title, fontsize=11)
    ax.set_xlabel('Time relative to event (s)')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

fig, axs = plt.subplots(2, 2, figsize=(9, 6.6))
ax = axs[0, 0]; band(ax, AC, COL_UNASSISTED, '-', 'Aha'); band(ax, RC, '0.45', (0, (4, 2)), 'Non-insight')
deco(ax, 'Unassisted'); ax.set_ylabel('F3/C3/P3 amplitude ($\\mu$V)'); ax.legend(frameon=False, fontsize=8.5, loc='lower left')
ax = axs[0, 1]; band(ax, AT, COL_LLM, '-', 'Aha'); band(ax, RT, '0.45', (0, (4, 2)), 'Non-insight')
deco(ax, 'LLM-assisted'); ax.legend(frameon=False, fontsize=8.5, loc='lower left')
yl = (min(axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0]), max(axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1]))
for x in axs[0]:
    x.set_ylim(yl)
ax = axs[1, 0]
band(ax, AC - AT, 'k', '-', 'Aha epochs'); band(ax, RC - RT, '0.45', (0, (4, 2)), 'Non-insight epochs')
deco(ax, 'Condition difference (Unassisted $-$ LLM-assisted)'); ax.set_ylabel('Difference ($\\mu$V)')
ax.legend(frameon=False, fontsize=8.5, loc='lower right')
ca = S['C_a']; ah = S['aha_reference']
ax.text(0.02, 0.97, f"Window mean, Wilcoxon (N = {N}):\nnon-insight W = {ca['W']:g}, p = {ca['p']:.3f}".replace('0.', '.'),
        transform=ax.transAxes, va='top', fontsize=8.5)
ax = axs[1, 1]
md = np.genfromtxt(f'{OUT}/07_noninsight_draws.csv', delimiter=',', names=True)['mean_diff_U_minus_LLM']
ax.hist(md, bins=40, color='0.7', edgecolor='white', lw=0.4)
ax.axvline(ah and S['C_b']['aha_mean_diff'], color='k', lw=1.5)
ax.text(S['C_b']['aha_mean_diff'], ax.get_ylim()[1] * 0.92, ' aha', fontsize=9, va='top')
ax.set_xlabel('ROI window difference, Unassisted $-$ LLM ($\\mu$V)'); ax.set_ylabel('Number of draws')
ax.set_title('Matched-count non-insight draws (1,000)', fontsize=11)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
cb = S['C_b']
ax.text(0.98, 0.97, f"p < .05 in {cb['pct_p_lt_05']:.1f}% of draws", transform=ax.transAxes, ha='right', va='top', fontsize=8.5)
for x, lab in zip(axs.flat, 'ABCD'):
    x.text(-0.14, 1.06, lab, transform=x.transAxes, fontsize=15, fontweight='bold')
fig.text(0.5, -0.01, 'Grey band: cluster window ($-450$ to $-297$ ms, from the insight-locked cluster analysis; not re-selected). Lines: mean across N = %d participants; shading: $\\pm$ SEM.' % N,
         ha='center', fontsize=8.5, color='0.3')
fig.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(f'{FIGDIR}/fig_noninsight_control.{ext}', dpi=300, bbox_inches='tight')
print('ok')
