"""Task 5: behavioural figure. (A) aha count, (B) aha rate per 10 min (09_aha_rate.py), (C) time on task, per participant.

Input : outputs/02_aha_counts_per_participant.csv, outputs/02_time_on_task_per_participant.csv,
        outputs/02_counts_time_summary.json
Output: Figures/v04/fig_behaviour_aha_time.{png,pdf}
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import *

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 10, 'pdf.fonttype': 42})
rng = np.random.default_rng(0)
cnt = pd.read_csv(f'{OUT}/02_aha_counts_per_participant.csv', index_col=0)
tim = pd.read_csv(f'{OUT}/02_time_on_task_per_participant.csv', index_col=0)
S = json.load(open(f'{OUT}/02_counts_time_summary.json'))
w_cnt = S['wilcoxon_counts_all19']['wilcox|two-sided|auto|corr=False']
w_min = S['wilcoxon_minutes_orig_all19']
rate = pd.read_csv(f'{OUT}/09_aha_rate_per_participant.csv', index_col=0).loc[cnt.index]
R = json.load(open(f'{OUT}/09_aha_rate.json'))['all19']
assert (tim.index == cnt.index).all()


def fmt_p(p):
    return f'p = {p:.4f}'.replace('0.', '.', 1) if p >= 0.001 else 'p < .001'


def panel(ax, u, l, in15, ylabel, title, ann, integer):
    n = len(u)
    jit = rng.uniform(-0.06, 0.06, n) if integer else np.zeros(n)
    x0, x1 = 0 + jit, 1 + jit
    for i in range(n):
        ax.plot([x0[i], x1[i]], [u[i], l[i]], color='0.55', lw=0.8, ls='-' if in15[i] else (0, (3, 2)), zorder=2)
    ax.scatter(x0, u, s=22, color=COL_UNASSISTED, edgecolor='white', lw=0.6, zorder=3)
    ax.scatter(x1, l, s=22, color=COL_LLM, marker='s', edgecolor='white', lw=0.6, zorder=3)
    bp = ax.boxplot([u, l], positions=[-0.32, 1.32], widths=0.18, patch_artist=True, showfliers=False,
                    medianprops=dict(color='k', lw=1.2), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, col in zip(bp['boxes'], [COL_UNASSISTED, COL_LLM]):
        patch.set_facecolor(col); patch.set_alpha(0.35); patch.set_edgecolor(col)
    for xp, v, col, mk in [(-0.32, u, COL_UNASSISTED, 'o'), (1.32, l, COL_LLM, 's')]:
        ax.scatter([xp], [np.mean(v)], marker='D', s=34, color='white', edgecolor='k', lw=1.0, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Unassisted', 'LLM-assisted'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='0.9', lw=0.6); ax.set_axisbelow(True)
    ax.text(0.5, 1.0, ann, transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='0.2')

fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.3))
in15 = cnt.in_N15.values.astype(bool)
panel(axs[0], cnt.control.values, cnt.treatment.values, in15, 'Confirmed aha moments per participant',
      '', f'Wilcoxon signed-rank, N = 19: W = {w_cnt[0]:g}, {fmt_p(w_cnt[1])}', True)
panel(axs[1], rate.rate_U.values, rate.rate_L.values, in15, 'Aha moments per 10 min of task time',
      '', f'Wilcoxon signed-rank, N = 19: W = {R["W"]:g}, {fmt_p(R["p"])}', False)
panel(axs[2], tim.control_min_orig.values, tim.treatment_min_orig.values, in15, 'Time on task per participant (min, 2 tasks)',
      '', f'Wilcoxon signed-rank, N = 19: W = {w_min["W"]:g}, {fmt_p(w_min["p"])}', False)
axs[0].set_ylim(-0.8, None)
axs[1].set_ylim(-0.2, None)
for ax, lab in zip(axs, 'ABC'):
    ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=15, fontweight='bold')
handles = [Line2D([], [], color='0.55', lw=0.8, label='Participant in N = 15 ERP sample'),
           Line2D([], [], color='0.55', lw=0.8, ls=(0, (3, 2)), label='Not in ERP sample (0 ahas in a condition)'),
           Line2D([], [], marker='D', color='white', markeredgecolor='k', ls='', label='Condition mean; box = median/IQR')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
for ext in ['png', 'pdf']:
    fig.savefig(f'{FIGDIR}/fig_behaviour_aha_time.{ext}', dpi=300, bbox_inches='tight')
print('counts W/p', w_cnt, 'minutes', w_min)
