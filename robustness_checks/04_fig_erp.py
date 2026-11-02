"""Task 4: new Figure 1 (participant-level, N = 15).

Input : outputs/03_participant_erps.npz (participant ERPs from 03_erp_cluster.py)
Output: Figures/v04/fig1_erp_cluster_participant_level.{png,pdf}; outputs/04_fig_erp_values.json
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import mne
from common import *

mne.set_log_level('ERROR')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 10, 'axes.linewidth': 0.8,
                     'pdf.fonttype': 42})

z = np.load(f'{OUT}/03_participant_erps.npz')
C, T, times, chans, pids = z['control'] * 1e6, z['treatment'] * 1e6, z['times'], list(z['chans']), list(z['pids'])
N = C.shape[0]
win = (times >= -0.4501) & (times <= -0.2966)
scalp = [chans.index(c) for c in SCALP_CHANS]

# montage/positions from one recording (EEGLAB Standard-10-20-Cap25 locations stored in the .set)
raw = read_raw(list_recordings()[0]['path'], preload=False)
info = raw.copy().pick(SCALP_CHANS).info

topo_u = C[:, scalp][:, :, win].mean((0, 2))
topo_l = T[:, scalp][:, :, win].mean((0, 2))
topo_d = topo_u - topo_l
vmax = float(np.ceil(np.max(np.abs(np.r_[topo_u, topo_l, topo_d]))))

fig = plt.figure(figsize=(11, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.32)
gtop = gs[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.15)
gbot = gs[1].subgridspec(1, 3, wspace=0.22)
titles = ['Unassisted', 'LLM-assisted', 'Difference\n(Unassisted $-$ LLM-assisted)', 'Difference\n(cluster channels)']
data = [topo_u, topo_l, topo_d, topo_d]
mask = np.array([c in ('F3', 'C3', 'P3') for c in SCALP_CHANS])
for k in range(4):
    ax = fig.add_subplot(gtop[0, k])
    kw = dict(mask=mask, mask_params=dict(marker='o', markerfacecolor='white', markeredgecolor='k',
                                          linewidth=0.8, markersize=15)) if k == 3 else {}
    im, _ = mne.viz.plot_topomap(data[k], info, axes=ax, show=False, cmap='RdBu_r', vlim=(-vmax, vmax),
                                 contours=6, sensors=True, names=SCALP_CHANS if k == 3 else None, **kw)
    if k == 3:
        for t in ax.texts:
            t.set_fontsize(6.5)
    ax.set_title(titles[k], fontsize=11)
cax = fig.add_subplot(gtop[0, 4])
cb = fig.colorbar(im, cax=cax)
cb.set_label('Amplitude ($\\mu$V)')
fig.text(0.07, 0.93, 'A', fontsize=16, fontweight='bold')
fig.text(0.07, 0.47, 'B', fontsize=16, fontweight='bold')
fig.text(0.5, 0.505, 'Mean over $-450$ to $-297$ ms; N = %d participants' % N, ha='center', fontsize=9.5, color='0.25')

vals = {}
ylo, yhi = np.inf, -np.inf
axes = []
for k, ch in enumerate(['F3', 'C3', 'P3']):
    ax = fig.add_subplot(gbot[0, k]); axes.append(ax)
    i = chans.index(ch)
    ax.axvspan(times[win][0], times[win][-1], color='0.85', lw=0, zorder=0)
    ax.axhline(0, color='0.3', lw=0.6, zorder=1)
    ax.axvline(0, color='k', ls='--', lw=0.9, zorder=1)
    for X, col, lab in [(C, COL_UNASSISTED, 'Unassisted'), (T, COL_LLM, 'LLM-assisted')]:
        m = X[:, i].mean(0); se = X[:, i].std(0, ddof=1) / np.sqrt(N)
        ax.fill_between(times, m - se, m + se, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(times, m, color=col, lw=1.5, zorder=3)
        ylo, yhi = min(ylo, (m - se).min()), max(yhi, (m + se).max())
        vals[f'{ch}_{lab}_window_mean_uV'] = float(X[:, i][:, win].mean())
    ax.set_title(ch, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel('Time relative to insight (s)')
    if k == 0:
        ax.set_ylabel('Amplitude ($\\mu$V)')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
pad = 0.08 * (yhi - ylo)
for ax in axes:
    ax.set_ylim(ylo - pad, yhi + pad)
handles = [Patch(color='0.85', label='Cluster window ($-450$ to $-297$ ms)'),
           Line2D([], [], color=COL_UNASSISTED, lw=1.5, label=f'Unassisted (N = {N})'),
           Line2D([], [], color=COL_LLM, lw=1.5, label=f'LLM-assisted (N = {N})')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.045))
fig.text(0.5, -0.07, 'Lines: mean across participants; shading: $\\pm$ SEM across participants', ha='center', fontsize=8.5, color='0.3')

for ext in ['png', 'pdf']:
    fig.savefig(f'{FIGDIR}/fig1_erp_cluster_participant_level.{ext}', dpi=300, bbox_inches='tight')
vals.update(dict(N=N, vlim_uV=vmax, topo_unassisted=dict(zip(SCALP_CHANS, map(float, topo_u))),
                 topo_llm=dict(zip(SCALP_CHANS, map(float, topo_l))),
                 topo_diff=dict(zip(SCALP_CHANS, map(float, topo_d))),
                 roi_unassisted=float(C[:, [chans.index(c) for c in ('F3', 'C3', 'P3')]][:, :, win].mean()),
                 roi_llm=float(T[:, [chans.index(c) for c in ('F3', 'C3', 'P3')]][:, :, win].mean())))
json.dump(vals, open(f'{OUT}/04_fig_erp_values.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in vals.items() if not k.startswith('topo')}, indent=1))
