"""Three-region summary of the dDTF result (paper v08, Table 2): one arrow per pathway that is lower with the
LLM after FDR correction (q < .05), labelled with its percent change. Pathways that do not pass are not drawn.
Theta, alpha and beta use separate colours and separate lanes. Opposite directions between the same two regions
bow to opposite sides, so labels never sit on each other.

Usage: python v8_ddtf_schematic.py tests.csv out.png      (tests.csv from v8_her_ddtf.py)
"""
import math
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path

tests, out = sys.argv[1], sys.argv[2]
t = pd.read_csv(tests)
parts = t.feature.str.extract(r'^(\w+) (\w+)->(\w+) dDTF$')
t['band'], t['src'], t['dst'] = parts[0], parts[1], parts[2]
t['pct'] = 100 * (t.mean_treatment - t.mean_control) / t.mean_control
sig = t[t.p_fdr_all < 0.05]

ROIS = {'Frontal': (0.50, 0.80), 'Parietal': (0.17, 0.30), 'Occipital': (0.83, 0.30)}
LABEL_SIDE = {'Frontal': 'above', 'Parietal': 'below', 'Occipital': 'below'}
BANDS = {'Theta': ('4-8 Hz', '#D55E00', 0.12), 'Alpha': ('8-13 Hz', '#0072B2', 0.25),
         'Beta': ('13-30 Hz', '#009E73', 0.38)}   # colour, lane (bow of the curve)
NODE_R = 0.085

fig, ax = plt.subplots(figsize=(7.4, 6.2), dpi=220)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.02, 0.98)
ax.set_aspect('equal')
ax.axis('off')


def arrow(a, b, band, pct):
    (x1, y1), (x2, y2) = ROIS[a], ROIS[b]
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    px, py = uy, -ux                      # right-hand side of the direction of flow
    _, color, lane = BANDS[band]
    sx, sy = x1 + ux * NODE_R, y1 + uy * NODE_R
    ex, ey = x2 - ux * NODE_R, y2 - uy * NODE_R
    mx, my = (sx + ex) / 2 + px * lane, (sy + ey) / 2 + py * lane
    width = 1.8 + 6.5 * abs(pct) / 100
    ax.add_patch(PathPatch(Path([(sx, sy), (mx, my), (ex, ey)], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                           facecolor='none', edgecolor=color, lw=width, zorder=2, capstyle='round'))
    tl = math.hypot(ex - mx, ey - my)
    ax.add_patch(FancyArrowPatch((ex - (ex - mx) / tl * 0.001, ey - (ey - my) / tl * 0.001), (ex, ey),
                                 arrowstyle='-|>', mutation_scale=20, color=color, lw=width,
                                 shrinkA=0, shrinkB=0, zorder=3))
    bx, by = 0.25 * sx + 0.5 * mx + 0.25 * ex, 0.25 * sy + 0.5 * my + 0.25 * ey
    ax.text(bx, by, f'{pct:+.0f}%', ha='center', va='center', fontsize=10, fontweight='bold', color=color,
            zorder=5, bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=color, lw=1.0))


for _, r in sig.sort_values('pct').iterrows():
    arrow(r.src, r.dst, r.band, r.pct)

for name, (x, y) in ROIS.items():
    ax.add_patch(Circle((x, y), NODE_R, facecolor='#f7f7f7', edgecolor='#1a1a1a', lw=2.0, zorder=4))
    ax.text(x, y, name[0], ha='center', va='center', fontsize=24, fontweight='bold', color='#1a1a1a', zorder=5)
    if LABEL_SIDE[name] == 'above':
        ax.text(x, y + NODE_R + 0.035, name, ha='center', va='bottom', fontsize=12, color='#333', zorder=5)
    else:
        ax.text(x, y - NODE_R - 0.03, name, ha='center', va='top', fontsize=12, color='#333', zorder=5)

ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=f'{b} ({hz})') for b, (hz, c, _) in BANDS.items()],
          loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=12, frameon=False, handlelength=2.4)
plt.savefig(out, dpi=220, bbox_inches='tight', facecolor='white', pad_inches=0.15)
print('wrote', out, len(sig), 'arrows')
