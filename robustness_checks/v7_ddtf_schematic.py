"""Clean 3-ROI schematic of the Option B Wallas-stage dDTF result (Table 2, v07).
Design rules:
  - Only significant pathways (FDR q < .05) are drawn solid; non-sig drawn as a
    very faint dotted line with no label.
  - Each arrow is a smooth cubic curve. Opposite directions between the same
    pair of nodes curve to opposite sides, so labels never collide.
  - Alpha and theta are offset to opposite sides of each path, and get
    different colours. One short label per arrow (percent change only).
  - FDR values live in the caption, not on the figure, to keep it clean.
Produces: Sections/Figures/v04/fig_ddtf_wallas_schematic.png
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import RESULTS  # noqa: E402
import os, sys  # noqa: E402
import os
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D


J = json.load(open(os.path.join(RESULTS, 'optB_5_ddtf_wallas.json')))
rows = J['stats']['wallas_mean']['stats']

ROIS = {
    'Frontal':   (0.50, 0.78),
    'Parietal':  (0.19, 0.33),
    'Occipital': (0.81, 0.33),
}
# Where to place each ROI name relative to its node (so arrow heads never cover text)
LABEL_SIDE = {'Frontal': 'above', 'Parietal': 'below', 'Occipital': 'below'}
BAND_COLOR = {'Alpha': '#1f6fd1', 'Theta': '#d62728'}
NODE_FACE = '#f7f7f7'
NODE_EDGE = '#1a1a1a'
SIG = 0.05
NODE_R = 0.085


def parse(pw):
    band, path = pw.split(' ', 1)
    a, b = path.split('->')
    return band, a, b


fig, ax = plt.subplots(figsize=(7.4, 5.9), dpi=220)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.00, 0.96)
ax.set_aspect('equal')
ax.axis('off')


def unit(dx, dy):
    L = math.hypot(dx, dy)
    return dx / L, dy / L, L


def _canonical(a, b):
    """Canonical ordering per pair, picked so the SIGNIFICANT direction for
    each pair sits on the OUTER lane (less crowded). In this data every
    frontal flow is a sink, so canonical sources rank Parietal, then
    Occipital, then Frontal."""
    rank = {'Parietal': 0, 'Occipital': 1, 'Frontal': 2}
    if rank[a] < rank[b]:
        return (a, b, True)
    return (b, a, False)


def curved_arrow(a, b, band, pct, sig):
    """One arrow A->B. Layout:
      - For a chord INVOLVING Frontal (P-F or O-F): both bands bow to the
        OUTSIDE of the triangle; alpha uses the outer radius, theta the
        inner radius (closer to the chord). This keeps all F-pair labels
        outside the triangle, so they never collide with P-O labels.
      - For the P-O chord: alpha bows ABOVE the chord, theta bows BELOW.
        Within each side, the canonical-forward direction uses the outer
        radius and the reverse direction the inner.
    Result: eight non-overlapping lanes, eight non-colliding labels."""
    (x1, y1), (x2, y2) = ROIS[a], ROIS[b]
    dx, dy = x2 - x1, y2 - y1
    ux, uy, L = unit(dx, dy)

    sx, sy = x1 + ux * NODE_R, y1 + uy * NODE_R
    ex, ey = x2 - ux * NODE_R, y2 - uy * NODE_R

    ca, cb, forward = _canonical(a, b)
    cx1, cy1 = ROIS[ca]; cx2, cy2 = ROIS[cb]
    cux, cuy, _ = unit(cx2 - cx1, cy2 - cy1)
    canon_px, canon_py = -cuy, cux  # left-hand perpendicular of canonical chord

    involves_F = ('Frontal' in (a, b))
    if involves_F:
        # Push the perpendicular AWAY from the triangle centroid for every
        # F-pair arrow. Side is fixed per pair, so alpha and theta share it.
        centroid = (sum(x for x, _ in ROIS.values()) / 3,
                    sum(y for _, y in ROIS.values()) / 3)
        chord_mid = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
        out_vec = (chord_mid[0] - centroid[0], chord_mid[1] - centroid[1])
        if canon_px * out_vec[0] + canon_py * out_vec[1] < 0:
            canon_px, canon_py = -canon_px, -canon_py
        px, py = canon_px, canon_py
        # Alpha gets the outer radius, theta the inner radius. All F-pair
        # labels end up on the OUTSIDE of the triangle.
        outer, inner = 0.33, 0.10
        curve_mag = outer if band == 'Alpha' else inner
    else:
        # P-O chord: alpha above, theta below; outer=forward, inner=reverse.
        band_side = 1.0 if band == 'Alpha' else -1.0
        px, py = canon_px * band_side, canon_py * band_side
        outer, inner = 0.24, 0.09
        curve_mag = outer if forward else inner

    mx = (sx + ex) / 2 + px * curve_mag
    my = (sy + ey) / 2 + py * curve_mag

    if sig:
        width = 1.8 + 6.5 * (abs(pct) / 100.0)
        color = BAND_COLOR[band]
        alpha = 1.0
        linestyle = '-'
    else:
        width = 1.0
        color = '#cfcfcf'
        alpha = 0.5
        linestyle = (0, (1, 3))

    path = Path([(sx, sy), (mx, my), (ex, ey)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, facecolor='none', edgecolor=color, lw=width,
                           linestyle=linestyle, alpha=alpha, zorder=2,
                           capstyle='round'))

    # Arrow head at the end, oriented along the local tangent
    if sig:
        tdx, tdy, _ = unit(ex - mx, ey - my)
        head = FancyArrowPatch((ex - tdx * 0.001, ey - tdy * 0.001),
                               (ex, ey),
                               arrowstyle='-|>', mutation_scale=20,
                               color=color, lw=width, alpha=alpha,
                               shrinkA=0, shrinkB=0, zorder=3)
        ax.add_patch(head)

    # Place each label directly ON its arc apex (zero perpendicular push), with
    # a solid white box: the label sits on top of its arrow so the reader cannot
    # misattribute it to a neighbouring one.
    if sig:
        bx = 0.25 * sx + 0.5 * mx + 0.25 * ex
        by = 0.25 * sy + 0.5 * my + 0.25 * ey
        ax.text(bx, by, f"{pct:+.0f}%",
                ha='center', va='center',
                fontsize=10, fontweight='bold', color=color, zorder=5,
                bbox=dict(boxstyle='round,pad=0.22', fc='white',
                          ec=color, lw=1.0, alpha=1.0))


# Non-significant first (behind), then significant on top
for row in sorted(rows, key=lambda r: r['q_paper'] < SIG):
    band, a, b = parse(row['pathway'])
    sig = row['q_paper'] < SIG
    curved_arrow(a, b, band, row['change_pct'], sig)

# Nodes on top of arrows
for name, (x, y) in ROIS.items():
    ax.add_patch(Circle((x, y), NODE_R, facecolor=NODE_FACE,
                        edgecolor=NODE_EDGE, lw=2.0, zorder=4))
    ax.text(x, y, name[0], ha='center', va='center',
            fontsize=24, fontweight='bold', color=NODE_EDGE, zorder=5)
    if LABEL_SIDE[name] == 'above':
        ax.text(x, y + NODE_R + 0.035, name, ha='center', va='bottom',
                fontsize=12, color='#333', zorder=5)
    else:
        ax.text(x, y - NODE_R - 0.090, name, ha='center', va='top',
                fontsize=12, color='#333', zorder=5)

# Legend
leg_handles = [
    Line2D([0], [0], color=BAND_COLOR['Alpha'], lw=3, label='Alpha (8-13 Hz)'),
    Line2D([0], [0], color=BAND_COLOR['Theta'], lw=3, label='Theta (4-8 Hz)'),
]
ax.legend(handles=leg_handles, loc='lower center',
          bbox_to_anchor=(0.5, -0.015), ncol=2, fontsize=12,
          frameon=False, handlelength=2.4)

# no title or subtitle; the LaTeX figure caption carries that

out = os.path.join(RESULTS, 'fig_ddtf_wallas_schematic.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=220, bbox_inches='tight', facecolor='white', pad_inches=0.15)
print('wrote', out)
