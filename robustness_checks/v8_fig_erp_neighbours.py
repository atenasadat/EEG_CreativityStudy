"""Appendix figure: the channel neighbours used by the ERP cluster test (v08). Lines are neighbour links; filled
circles are the channels of the pre-aha cluster. Input: OUT_DIR/v8_erp_spatial.json, v8_erp_neighbours.json (from v8_erp_spatial.py).
Output: OUT_DIR/figures/fig_erp_neighbours.{png,pdf}"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import *

R = json.load(open(f'{OUT}/v8_erp_spatial.json'))['tests']
NB = json.load(open(f'{OUT}/v8_erp_neighbours.json'))['neighbours']
raw0 = read_raw(list_recordings()[0]['path'], preload=False).pick(SCALP_CHANS)
pos = {ch['ch_name']: np.array(ch['loc'][:3]) for ch in raw0.info['chs']}
ctr = np.mean(list(pos.values()), axis=0) * np.array([1, 1, 0])


def proj(p):  # azimuthal equidistant projection, nose up
    p = p - ctr
    r = np.linalg.norm(p); th = np.arccos(np.clip(p[2] / r, -1, 1)); ph = np.arctan2(p[1], p[0])
    return np.array([th * np.cos(ph), th * np.sin(ph)])


xy = {c: proj(p) for c, p in pos.items()}
a = np.arctan2(xy['Fz'][1], xy['Fz'][0]) - np.pi / 2
rot = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])
xy = {c: rot @ v for c, v in xy.items()}
rad = max(np.linalg.norm(v) for v in xy.values()) * 1.12
order = {c: [SCALP_CHANS[j] for j in (i - 1, i + 1) if 0 <= j < len(SCALP_CHANS)] for i, c in enumerate(SCALP_CHANS)}
INK, MUTED, EDGE, ACC = '#1f1f1f', '#6b6b6b', '#b8b8b8', '#2a6fdb'


def f(p):
    return f'{p:.3f}'.lstrip('0')


panels = [(NB, R['spatial (Delaunay) | pre -0.5..0'], R['spatial (Delaunay) | full -0.5..0.5'], '')]
fig, axes = plt.subplots(1, 1, figsize=(4.6, 4.9))
axes = [axes]
for ax, (nb, pre, full, title) in zip(axes, panels):
    clus = set(pre['channels'])
    deg = np.mean([len(v) for v in nb.values()])
    ax.add_patch(plt.Circle((0, 0), rad, fill=False, lw=1.2, color=MUTED))
    ax.plot([-0.12 * rad, 0, 0.12 * rad], [rad * 0.99, rad * 1.1, rad * 0.99], color=MUTED, lw=1.2)
    done = set()
    for c, vs in nb.items():
        for v in vs:
            e = tuple(sorted((c, v)))
            if e in done:
                continue
            done.add(e)
            both = c in clus and v in clus
            ax.plot(*zip(xy[c], xy[v]), color=ACC if both else EDGE, lw=2.2 if both else 1.0, zorder=1,
                    solid_capstyle='round')
    for c, v in xy.items():
        ax.scatter(*v, s=150, color=ACC if c in clus else 'white', edgecolor=ACC if c in clus else INK,
                   lw=0.8, zorder=2)
        ax.text(v[0] + 0.06 * rad, v[1] - 0.06 * rad, c, ha='left', va='top', fontsize=8, color=INK)
    ax.set_title(f'Delaunay triangulation of the electrode positions\n{deg:.1f} neighbours per channel on average',
                 fontsize=9.5, color=INK, loc='left')
    ax.set_xlim(-rad * 1.08, rad * 1.08); ax.set_ylim(-rad * 1.08, rad * 1.18)
    ax.set_aspect('equal'); ax.axis('off')
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02)
for ext in ('png', 'pdf'):
    plt.savefig(f'{FIGDIR}/fig_erp_neighbours.{ext}', dpi=220)
print('saved', FIGDIR)
