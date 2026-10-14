"""Main-paper figure support: Kosmyna-style EEG visualizations.

This script renders the dDTF brain-network figure, whole-session band-power
topomaps, and whole-session coherence networks used or adapted in main.tex.

Kosmyna-style ("Your Brain on ChatGPT") renderings of whole-session band power
and coherence networks.

Outputs (in ./kosmyna_figures):
  topomaps_band_power.png          rows: Unassisted / LLM-Assisted / p values, cols: bands (top view)
  network_Unassisted_<band>.png    left / top / right views of the condition's coherence network
  network_LLM-Assisted_<band>.png
  network_comparison_<band>.png    Kosmyna Fig. 2 layout: Unassisted, LLM-Assisted, P values,
                                   Unassisted significant, LLM-Assisted significant

Per-subject measures are cached to kosmyna_figures/subject_measures.npz; pass
--recompute to rebuild them from the raw EEG.

The Wallas-segment figures are drawn from wallas_analysis/wallas_mapping.ipynb, which
computes the results and passes them in (nothing is read from disk):
  plot_ddtf_networks(tests, question, surface)  -> ../wallas_analysis/figures/
      q<n>_ddtf_kosmyna_networks.{png,svg} (surface="head") or q<n>_ddtf_brain_networks
  plot_q2_alpha_topomaps(pid_diff)              -> ../wallas_analysis/figures/q2_alpha_topomaps
"""
import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import mne
import numpy as np
from scipy.interpolate import RBFInterpolator

from wholesession_shared import (
    BANDS,
    coherence_matrices,
    edge_list,
    ensure_dir,
    iter_subject_raws,
    mean_psd_by_band,
    rs,
)
from wholesession_ddtf import ROIS


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "kosmyna_figures")
CACHE_PATH = os.path.join(OUTPUT_DIR, "subject_measures.npz")
COND_LABELS = {"Control": "Unassisted", "Treatment": "LLM-Assisted"}
VIEWS = ["left", "top", "right"]
ALPHA_SIG = 0.05

# Kosmyna palettes: jet for strength, blue -> green -> orange for significance
STRENGTH_CMAP = plt.cm.jet
P_CMAP = LinearSegmentedColormap.from_list(
    "kosmyna_p", ["#3B8BEB", "#5FD39A", "#B9E05A", "#F4A142", "#E8603C"]
)
HEAD_FILL = "#E4E4E4"
HEAD_EDGE = "#B5B5B5"
STANDARD_19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
               "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]

# Wallas-segment dDTF test tables from wallas_mapping.ipynb (q1_ddtf_tests,
# q2_ddtf_tests): one row per directed ROI pathway with its group means and FDR p. Each
# question compares two states with a mixed-model LR test of state + state x Wallas step.
#   q1: Unassisted vs LLM-Assisted sessions (section 12)
#   q2: within LLM-Assisted sessions, No AI interaction vs AI interaction segments
#       (inquire / adoptCopilot), the Q2 split of section 13
WALLAS_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                              "wallas_analysis", "figures")
DDTF_QUESTIONS = {
    "q1": {"means": ("mean_control", "mean_treatment"),
           "labels": ("Unassisted", "LLM-Assisted")},
    "q2": {"means": ("mean_no_ai", "mean_ai"),
           "labels": ("No AI interaction", "AI interaction")},
}
DDTF_TOP_EDGES = 10


def ddtf_rows(tests, question):
    """[(band, source, target, mean state A, mean state B, FDR p)] for every pathway."""
    parts = tests.feature.str.extract(r"^(\w+) (\w+)->(\w+) dDTF$")
    a, b = DDTF_QUESTIONS[question]["means"]
    return list(zip(parts[0], parts[1], parts[2], tests[a], tests[b], tests.p_fdr_all))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA_SIG else ""


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def compute_subject_measures(max_subjects=None):
    pids, ch_names = [], None
    coh = {cond: {band: [] for band in BANDS} for cond in COND_LABELS}
    power = {cond: {band: [] for band in BANDS} for cond in COND_LABELS}
    for subj in iter_subject_raws(max_subjects=max_subjects):
        ch_names = subj["ch_names"]
        pids.append(subj["Pid"])
        for cond in COND_LABELS:
            mats = coherence_matrices(subj[cond], ch_names)
            bp, _ = mean_psd_by_band(subj[cond], ch_names)
            for band in BANDS:
                coh[cond][band].append(mats[band])
                power[cond][band].append(bp[band])
        print("  coherence + band power extracted", flush=True)

    arrays = {"pids": np.array(pids), "ch_names": np.array(ch_names)}
    for cond in COND_LABELS:
        for band in BANDS:
            arrays[f"coh_{cond}_{band}"] = np.asarray(coh[cond][band])
            arrays[f"pow_{cond}_{band}"] = np.asarray(power[cond][band])
    ensure_dir(OUTPUT_DIR)
    np.savez(CACHE_PATH, **arrays)
    return dict(np.load(CACHE_PATH))


def load_measures(recompute=False, max_subjects=None):
    if recompute or not os.path.exists(CACHE_PATH):
        return compute_subject_measures(max_subjects)
    return dict(np.load(CACHE_PATH))


def paired_p(control, treatment):
    """Exact paired Wilcoxon p for each column (subjects x features)."""
    return np.array([
        rs.exact_wilcoxon(control[:, k], treatment[:, k])["p"]
        for k in range(control.shape[1])
    ])


# --------------------------------------------------------------------------
# Geometry: 3D montage -> orthographic views of a glass head
# --------------------------------------------------------------------------
def sensor_xyz(ch_names):
    montage = mne.channels.make_standard_montage("standard_1020")
    pos = montage.get_positions()["ch_pos"]
    xyz = np.array([pos[ch] for ch in ch_names])
    xyz = xyz - np.array([0.0, -0.017, 0.0])  # roughly center the head sphere
    return xyz / np.max(np.linalg.norm(xyz, axis=1))


def project(xyz, view):
    """Orthographic projection; returns (2D coords, depth toward viewer)."""
    x, y, z = xyz.T
    if view == "top":
        return np.column_stack([x, y]), z
    if view == "left":   # viewer at -x, nose points left
        return np.column_stack([-y, z]), -x
    if view == "right":  # viewer at +x, nose points right
        return np.column_stack([y, z]), x
    raise ValueError(view)


def draw_head(ax, view):
    kw = dict(facecolor=HEAD_FILL, edgecolor=HEAD_EDGE, linewidth=1.2, zorder=0)
    if view == "top":
        ax.add_patch(Polygon([[-0.13, 1.02], [0.0, 1.17], [0.13, 1.02]], closed=True, **kw))
        for sx in (-1, 1):
            ax.add_patch(Ellipse((sx * 1.04, 0.0), 0.14, 0.32, **kw))
        ax.add_patch(Ellipse((0, 0), 2.08, 2.12, **kw))
        ax.add_patch(Ellipse((0, 0.08), 1.6, 1.55, facecolor="white", alpha=0.35,
                             linewidth=0, zorder=0))
        return
    s = -1 if view == "left" else 1  # face direction on the x axis
    face = np.array([
        [0.80, 0.55], [0.98, 0.30], [1.00, 0.05], [1.12, -0.18], [1.03, -0.25],
        [1.04, -0.38], [0.98, -0.48], [0.96, -0.62], [0.80, -0.80], [0.55, -0.85],
        [0.45, -1.00], [0.35, -1.30], [-0.40, -1.30], [-0.50, -0.82],
    ])
    face[:, 0] *= s
    ax.add_patch(Polygon(face, closed=False, **kw))
    ax.add_patch(Ellipse((-s * 0.04, 0.05), 2.12, 1.95, **kw))
    ax.add_patch(Ellipse((s * 0.0, -0.22), 0.16, 0.30, facecolor="#D2D2D2",
                         edgecolor=HEAD_EDGE, linewidth=1.0, zorder=0.5))
    ax.add_patch(Ellipse((-s * 0.1, 0.2), 1.6, 1.3, facecolor="white", alpha=0.35,
                         linewidth=0, zorder=0.6))


def finish_axis(ax, view):
    ax.set_aspect("equal")
    if view == "top":
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.2, 1.25)
    else:
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.38, 1.1)
    ax.axis("off")


def draw_sensors(ax, xy, depth, ch_names, label_size, view="top"):
    front = np.ones(len(depth), bool) if view == "top" else depth >= -0.05
    ax.scatter(xy[~front, 0], xy[~front, 1], s=8, c="white", edgecolors="#C8C8C8",
               linewidths=0.4, zorder=3, alpha=0.6)
    ax.scatter(xy[front, 0], xy[front, 1], s=14, c="white", edgecolors="#8A8A8A",
               linewidths=0.5, zorder=4)
    for k, ch in enumerate(ch_names):
        if not front[k]:
            continue
        ax.text(xy[k, 0], xy[k, 1] + 0.065, ch, fontsize=label_size, ha="center",
                va="bottom", color="#222222", zorder=6)


def draw_network(ax, view, xyz, ch_names, values, norm, cmap, keep=None, label_size=7):
    """values: (n_edges,) aligned with edge_list(ch_names); keep: optional bool mask."""
    draw_head(ax, view)
    xy, depth = project(xyz, view)
    edges = edge_list(ch_names)
    order = np.argsort(values)  # strongest drawn last (on top)
    segs, colors, widths = [], [], []
    for k in order:
        if keep is not None and not keep[k]:
            continue
        if not np.isfinite(values[k]):
            continue
        i, j = edges[k]
        frac = float(np.clip(norm(values[k]), 0, 1))
        segs.append([xy[i], xy[j]])
        rgba = list(cmap(frac))
        rgba[3] = 0.25 + 0.7 * frac
        colors.append(rgba)
        widths.append(0.4 + 2.2 * frac ** 1.5)
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=2,
                                     capstyle="round"))
    draw_sensors(ax, xy, depth, ch_names, label_size, view)
    finish_axis(ax, view)


def draw_scalp(ax, xyz, ch_names, values, norm, cmap, label_size=7, min_visible=None):
    """Top-view glass head with values interpolated over the scalp (thin-plate spline).
    Pixels below `min_visible` (in normalized units) are left transparent."""
    draw_head(ax, "top")
    xy, depth = project(xyz, "top")
    u, v = np.meshgrid(np.linspace(-1.02, 1.02, 260), np.linspace(-1.02, 1.02, 260))
    r_max = np.max(np.linalg.norm(xy, axis=1)) + 0.05
    inside = u ** 2 + v ** 2 <= r_max ** 2
    rbf = RBFInterpolator(xy, values, kernel="thin_plate_spline", smoothing=1e-3)
    img = np.full(u.shape, np.nan)
    # clip to the measured range so the spline cannot overshoot past edge sensors
    img[inside] = np.clip(rbf(np.column_stack([u[inside], v[inside]])),
                          np.min(values), np.max(values))
    if min_visible is not None:
        img[norm(img) < min_visible] = np.nan
    ax.imshow(np.ma.masked_invalid(img), extent=(-1.02, 1.02, -1.02, 1.02), origin="lower",
              cmap=cmap, norm=norm, alpha=0.85, zorder=1, interpolation="bilinear")
    draw_sensors(ax, xy, depth, ch_names, label_size, "top")
    finish_axis(ax, "top")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def edge_values(stack):
    """(subjects, n, n) -> (subjects, n_edges) for the upper triangle."""
    n = stack.shape[1]
    iu = np.triu_indices(n, k=1)
    return stack[:, iu[0], iu[1]]


def add_colorbar(fig, rect, cmap, low_label, high_label, title):
    cax = fig.add_axes(rect)
    grad = np.linspace(0, 1, 256)[:, None]
    cax.imshow(grad, aspect="auto", cmap=cmap, origin="lower")
    cax.set_xticks([])
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)
    cax.text(0.5, 1.02, high_label, transform=cax.transAxes, ha="center", va="bottom",
             fontsize=10)
    cax.text(0.5, -0.02, low_label, transform=cax.transAxes, ha="center", va="top",
             fontsize=10)
    cax.text(-0.6, 0.5, title, transform=cax.transAxes, ha="right", va="center",
             fontsize=14, fontweight="bold", rotation=90)


def p_norm():
    """Map p in [ALPHA_SIG, 1e-3] to [0, 1] on a log scale (1 = most significant)."""
    lo, hi = -np.log10(ALPHA_SIG), 3.0
    return lambda p: np.clip((-np.log10(np.maximum(p, 1e-12)) - lo) / (hi - lo), 0, 1)


def plot_condition_network(measures, ch_names, xyz, band, cond, norm):
    vals = edge_values(measures[f"coh_{cond}_{band}"]).mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=300,
                             gridspec_kw={"wspace": 0.02})
    for ax, view in zip(axes, VIEWS):
        draw_network(ax, view, xyz, ch_names, vals, norm, STRENGTH_CMAP, label_size=8)
    fig.text(0.02, 0.92, f"{COND_LABELS[cond]} — {band} coherence", fontsize=15,
             fontweight="bold", ha="left")
    add_colorbar(fig, [0.93, 0.15, 0.014, 0.65], STRENGTH_CMAP,
                 f"{norm.vmin:.2f}", f"{norm.vmax:.2f}", "")
    fig.text(0.955, 0.48, "Coherence", rotation=90, va="center", fontsize=12)
    fname = os.path.join(OUTPUT_DIR, f"network_{COND_LABELS[cond]}_{band}.png")
    fig.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_network_comparison(measures, ch_names, xyz, band, norm):
    ctrl = edge_values(measures[f"coh_Control_{band}"])
    trt = edge_values(measures[f"coh_Treatment_{band}"])
    p = paired_p(ctrl, trt)
    sig = p < ALPHA_SIG
    q = rs.fdr_bh(p)
    print(f"{band}: {sig.sum()} / {len(p)} edges p < {ALPHA_SIG} (uncorrected); "
          f"{int((q < ALPHA_SIG).sum())} survive FDR")
    pn = p_norm()
    p_score = np.where(sig, pn(p), np.nan)
    unit = Normalize(0, 1)

    rows = [
        ("Unassisted", ctrl.mean(0), norm, STRENGTH_CMAP, None),
        ("LLM-Assisted", trt.mean(0), norm, STRENGTH_CMAP, None),
        ("P values", p_score, unit, P_CMAP, sig),
        ("Unassisted\nsignificant", ctrl.mean(0), norm, STRENGTH_CMAP, sig),
        ("LLM-Assisted\nsignificant", trt.mean(0), norm, STRENGTH_CMAP, sig),
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 4.0 * len(rows)), dpi=300,
                             gridspec_kw={"wspace": 0.0, "hspace": 0.05, "left": 0.16,
                                          "right": 0.99, "top": 0.95, "bottom": 0.1})
    for r, (label, vals, nrm, cmap, keep) in enumerate(rows):
        for c, view in enumerate(VIEWS):
            draw_network(axes[r, c], view, xyz, ch_names, vals, nrm, cmap, keep=keep)
        pos = axes[r, 0].get_position()
        fig.text(0.015, (pos.y0 + pos.y1) / 2, label, fontsize=15, ha="left", va="center")
    fig.suptitle(f"{band}-band coherence networks (N = {ctrl.shape[0]})", fontsize=16,
                 fontweight="bold", y=0.975)
    for rect, cmap, lo, hi, caption in [
        ([0.25, 0.065, 0.5, 0.01], STRENGTH_CMAP, f"{norm.vmin:.2f}", f"{norm.vmax:.2f}",
         "Coherence (rows 1, 2, 4, 5)"),
        ([0.25, 0.025, 0.5, 0.01], P_CMAP, f"p = {ALPHA_SIG}", "p ≤ 0.001",
         "Paired Wilcoxon p, uncorrected (row 3; rows 4–5 show only p < 0.05 edges)"),
    ]:
        cax = fig.add_axes(rect)
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
        cax.set_xticks([])
        cax.set_yticks([])
        cax.text(-0.02, 0.5, lo, transform=cax.transAxes, ha="right", va="center", fontsize=11)
        cax.text(1.02, 0.5, hi, transform=cax.transAxes, ha="left", va="center", fontsize=11)
        cax.set_title(caption, fontsize=11, pad=4)
    fname = os.path.join(OUTPUT_DIR, f"network_comparison_{band}.png")
    fig.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_band_power(measures, ch_names, xyz):
    bands = list(BANDS)
    fig, axes = plt.subplots(3, len(bands), figsize=(3.3 * len(bands), 10.5), dpi=300,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.08})
    pn = p_norm()
    for c, band in enumerate(bands):
        # log10 power, shared 0-100% scale across conditions within a band
        ctrl = np.log10(measures[f"pow_Control_{band}"])
        trt = np.log10(measures[f"pow_Treatment_{band}"])
        mc, mt = ctrl.mean(0), trt.mean(0)
        lo, hi = min(mc.min(), mt.min()), max(mc.max(), mt.max())
        norm = Normalize(lo, hi)
        draw_scalp(axes[0, c], xyz, ch_names, mc, norm, STRENGTH_CMAP)
        draw_scalp(axes[1, c], xyz, ch_names, mt, norm, STRENGTH_CMAP)
        p = paired_p(ctrl, trt)
        q = rs.fdr_bh(p)
        print(f"{band} power: {int((p < ALPHA_SIG).sum())} channels p < {ALPHA_SIG}; "
              f"{int((q < ALPHA_SIG).sum())} survive FDR")
        draw_scalp(axes[2, c], xyz, ch_names, np.where(p < ALPHA_SIG, 0.05 + 0.95 * pn(p), 0.0),
                   Normalize(0, 1), P_CMAP, min_visible=0.04)
        axes[0, c].set_title(f"{band}\n[{BANDS[band][0]:.0f}–{BANDS[band][1]:.0f} Hz]",
                             fontsize=14, fontweight="bold")
    for r, label in enumerate(["Unassisted", "LLM-Assisted", "P values"]):
        axes[r, 0].text(-0.12, 0.5, label, transform=axes[r, 0].transAxes, fontsize=14,
                        fontweight="bold", rotation=90, ha="right", va="center")
    add_colorbar(fig, [0.925, 0.4, 0.012, 0.45], STRENGTH_CMAP, "lower\npower",
                 "higher\npower", "")
    add_colorbar(fig, [0.925, 0.13, 0.012, 0.2], P_CMAP, f"p = {ALPHA_SIG}", "p ≤ 0.001",
                 "")
    fig.text(0.965, 0.625, "log$_{10}$ band power\n(0–100% within band)", rotation=90,
             va="center", ha="center", fontsize=11)
    fig.text(0.965, 0.23, "Paired Wilcoxon p\n(uncorrected; blank = n.s.)", rotation=90, va="center",
             ha="center", fontsize=11)
    fname = os.path.join(OUTPUT_DIR, "topomaps_band_power.png")
    fig.savefig(fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fname}")


# --------------------------------------------------------------------------
# 3D surfaces for the ROI network figures: a glass head (Kosmyna Fig. 2 views) or the
# fsaverage cortex (lateral and dorsal views). Coordinates are fsaverage MRI space in metres
# (x right, y anterior, z up), the same space as the standard_1020 montage.
# --------------------------------------------------------------------------
LIGHT3D = np.array([-0.4, 0.5, 0.75])


class Scene:
    """A surface mesh plus how to frame and shade it.
    views: name -> (elevation, azimuth); draw(ax, view) renders the surface only."""

    def __init__(self, rr, tris, center, half_width, zoom, views, view_titles, shade,
                 faded_rgb=(0.6, 0.6, 0.6)):
        self.rr, self.tris = rr, tris
        self.faded_rgb = np.array([*faded_rgb, 1.0])  # what non-significant arcs fade toward
        self.center, self.half_width, self.zoom = np.asarray(center), half_width, zoom
        self.views, self.view_titles, self.shade = views, view_titles, shade
        tri = rr[tris]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        self.face_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    def view_dir(self, view):
        elev, azim = np.radians(self.views[view])
        return np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim),
                         np.sin(elev)])

    def draw(self, ax, view):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        rgba = self.shade(self, view)
        # matching edges hide the hairline seams matplotlib leaves between faces
        edges = rgba.copy() if rgba[:, 3].min() == 1 else "none"
        ax.add_collection3d(Poly3DCollection(self.rr[self.tris], facecolors=rgba,
                                             edgecolors=edges, linewidths=0.2))
        ax.set_proj_type("ortho")
        ax.view_init(*self.views[view])
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), self.center):
            setter(c - self.half_width, c + self.half_width)
        ax.set_box_aspect((1, 1, 1), zoom=self.zoom)
        ax.computed_zorder = False
        ax.axis("off")

    def near_side(self, points, view):
        """True for points on the half facing the viewer (all of them from above)."""
        if view == "top":
            return np.ones(len(points), bool)
        return (points - self.center) @ self.view_dir(view) > -0.01

    def onto_surface(self, points, outward=1.01):
        """Move each point along its ray from the centre to the outermost surface vertex
        on (nearly) that ray."""
        rel = self.rr - self.center
        radius = np.linalg.norm(rel, axis=1)
        dirs = rel / radius[:, None]
        out = []
        for p in np.atleast_2d(points):
            d = (p - self.center) / np.linalg.norm(p - self.center)
            cos = dirs @ d
            near = cos > np.max(cos) - 0.003
            k = np.flatnonzero(near)[np.argmax(radius[near])]
            out.append(self.center + d * radius[k] * outward)
        return np.array(out)


def shade_glass(scene, view):
    facing = np.abs(scene.face_normals @ scene.view_dir(view))
    light = np.clip(scene.face_normals @ (LIGHT3D / np.linalg.norm(LIGHT3D)), 0, 1)
    grey = np.clip(0.55 + 0.3 * light + 0.1 * facing, 0, 1)
    # glass: faces seen edge-on (silhouette) are more opaque than faces seen head-on
    return np.column_stack([grey, grey, grey, 0.16 + 0.3 * (1 - facing)])


def shade_cortex(scene, view):
    """Opaque grey cortex: sulci darker than gyri, lit from the viewer and slightly above."""
    light = scene.view_dir(view) + np.array([0.0, 0.0, 0.6])
    light /= np.linalg.norm(light)
    # smooth (per-vertex) lighting averaged onto faces
    vn = np.zeros_like(scene.rr)
    for k in range(3):
        np.add.at(vn, scene.tris[:, k], scene.face_normals)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    lam = np.clip(vn @ light, 0, 1)[scene.tris].mean(axis=1)
    base = np.where(scene.sulc[scene.tris].mean(axis=1) > 0, 0.68, 0.86)
    grey = np.clip(base * (0.62 + 0.45 * lam), 0, 1)
    return np.column_stack([grey, grey, grey, np.ones_like(grey)])


def head_scene():
    fs_dir = os.path.join(os.path.dirname(mne.__file__), "data", "fsaverage")
    surf = mne.read_bem_surfaces(os.path.join(fs_dir, "fsaverage-head.fif"), verbose=False)[0]
    views = {"right": (0, 0), "top": (90, -90), "oblique": (50, 35), "left": (0, 180)}
    titles = {"right": "Right", "top": "Top", "oblique": "Front-right oblique", "left": "Left"}
    return Scene(surf["rr"], surf["tris"], [0.0, -0.015, -0.03], 0.115, 1.15, views, titles,
                 shade_glass)


def _read_fs_surface(path):
    """FreeSurfer triangle surface (big-endian binary) without nibabel."""
    with open(path, "rb") as f:
        assert f.read(3) == b"\xff\xff\xfe", path
        f.readline()
        f.readline()
        n_vert, n_face = np.fromfile(f, ">i4", 2)
        rr = np.fromfile(f, ">f4", n_vert * 3).reshape(n_vert, 3).astype(float)
        tris = np.fromfile(f, ">i4", n_face * 3).reshape(n_face, 3)
    return rr, tris


def _read_fs_curv(path):
    with open(path, "rb") as f:
        assert f.read(3) == b"\xff\xff\xff", path
        n_vert = np.fromfile(f, ">i4", 3)[0]
        return np.fromfile(f, ">f4", n_vert).astype(float)


def brain_scene(ico=5):
    """fsaverage pial surface at icosahedral resolution `ico` (5 = fsaverage5): the first
    vertices of fsaverage are the ico subdivision, so they pair with the ico triangles."""
    from mne.surface import _get_ico_surface
    surf_dir = os.path.join(mne.datasets.fetch_fsaverage(verbose=False), "surf")
    ico_tris = _get_ico_surface(ico)["tris"]
    n = ico_tris.max() + 1
    rr, tris, sulc = [], [], []
    for h, hemi in enumerate(("lh", "rh")):
        hemi_rr, _ = _read_fs_surface(os.path.join(surf_dir, f"{hemi}.pial"))
        rr.append(hemi_rr[:n] / 1000.0)
        sulc.append(_read_fs_curv(os.path.join(surf_dir, f"{hemi}.sulc"))[:n])
        tris.append(ico_tris + h * n)
    rr = np.vstack(rr)
    center = (rr.min(axis=0) + rr.max(axis=0)) / 2
    half_width = 0.5 * np.max(rr.max(axis=0) - rr.min(axis=0)) * 1.08
    views = {"left": (0, 180), "top": (90, -90), "right": (0, 0)}
    titles = {"left": "Left", "top": "Top", "right": "Right"}
    # non-significant arcs fade toward a darker grey so they stay visible on the light cortex
    scene = Scene(rr, np.vstack(tris), center, half_width, 1.5, views, titles, shade_cortex,
                  faded_rgb=(0.35, 0.35, 0.35))
    scene.sulc = np.concatenate(sulc)
    return scene


class Arrow3D(FancyArrowPatch):
    def __init__(self, start, end, **kw):
        super().__init__((0, 0), (0, 0), **kw)
        self._verts3d = np.array([start, end]).T

    def do_3d_projection(self, renderer=None):
        from mpl_toolkits.mplot3d import proj3d
        xs, ys, zs = proj3d.proj_transform(*self._verts3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def arc3d(a, b, side, center, scale=1.0, n=40):
    """Quadratic arc from a to b. The two directions of a pair are separated both sideways
    (visible from above) and in height off the surface (visible in the profile views)."""
    mid = (a + b) / 2
    out = mid - center
    out /= np.linalg.norm(out)
    # sideways direction fixed per pair (not per direction) so `side` alone picks the side
    sideways = np.cross(b - a, out) * side
    sideways /= np.linalg.norm(sideways)
    length = np.linalg.norm(b - a)
    # short pairs need a wide bow to stay legible; frontal-occipital bows wider than
    # frontal-parietal so the two do not coincide from above
    # thresholds are for the head; `scale` shrinks them for smaller surfaces
    bend = 1.1 if length < 0.08 * scale else 0.45 if length > 0.2 * scale else 0.22
    lift = (0.45 if side > 0 else 0.08) * length
    ctrl = mid + out * lift + sideways * side * bend * length
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b


def sensor_points(scene, ch_names):
    """standard_1020 positions (already in fsaverage MRI space) placed on the surface."""
    pos = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    return scene.onto_surface(np.array([pos[ch] for ch in ch_names]))


def roi_nodes(scene, ch_names, xyz):
    """ROI name -> surface point straight out from the mean position of its sensors."""
    return {roi: scene.onto_surface(xyz[[ch_names.index(ch) for ch in chans]].mean(axis=0),
                                    outward=1.015)[0]
            for roi, chans in ROIS.items()}


def draw_roi_network(ax, view, scene, ch_names, xyz, nodes, arrows, norm, cmap):
    """arrows: [(source ROI, target ROI, value, FDR p)] drawn as directed 3D arcs over the
    scene's surface. FDR-significant pathways are drawn bold and starred; the rest thin and
    faded."""
    scene.draw(ax, view)
    c = scene.center
    front = scene.near_side(xyz, view)
    # non-significant first so significant arcs sit on top
    for src, tgt, value, q in sorted(arrows, key=lambda a: (a[3] < ALPHA_SIG, norm(a[2]))):
        frac = float(np.clip(norm(value), 0, 1))
        sig = q < ALPHA_SIG
        side = 1 if list(ROIS).index(src) < list(ROIS).index(tgt) else -1
        pts = arc3d(nodes[src], nodes[tgt], side, c, scene.half_width / 0.115)
        trim = slice(4, -5)  # keep the arc clear of the node markers
        if sig:
            color, width, alpha = cmap(frac), 1.8 + 3.0 * frac, 0.95
        else:  # faded toward grey
            color = 0.35 * np.array(cmap(frac)) + 0.65 * scene.faded_rgb
            width, alpha = 0.9 + 0.8 * frac, 0.55
        ax.plot(*pts[trim].T, color=color, linewidth=width, alpha=alpha,
                solid_capstyle="round", linestyle="-" if sig else (0, (3, 2)))
        ax.add_artist(Arrow3D(pts[-7], pts[-5], arrowstyle="-|>", color=color,
                              mutation_scale=(10 + 8 * frac) if sig else 8,
                              linewidth=width, alpha=alpha))
        if sig:
            mid = pts[len(pts) // 2]
            ax.text(*(c + (mid - c) * 1.06), stars(q), fontsize=13, fontweight="bold",
                    ha="center", va="center", color="#111111")
    ax.scatter(*xyz[front].T, s=22, c="white", edgecolors="#6A6A6A", linewidths=0.6,
               depthshade=False)
    for ch, p in zip(np.array(ch_names)[front], xyz[front]):
        ax.text(*(c + (p - c) * 1.1), ch, fontsize=11, ha="center", va="center",
                color="#222222")
    for roi, p in nodes.items():
        if not scene.near_side(p[None], view)[0]:
            continue
        ax.scatter(*p, s=190, c="white", edgecolors="#444444", linewidths=1.2, depthshade=False)
        ax.text(*p, roi[0], fontsize=10, fontweight="bold", ha="center", va="center",
                color="#111111")


def plot_ddtf_networks(tests, question="q1", surface="head"):
    scene = head_scene() if surface == "head" else brain_scene()
    out_name = f"{question}_ddtf_{'kosmyna' if surface == 'head' else 'brain'}_networks"
    label_a, label_b = DDTF_QUESTIONS[question]["labels"]
    ch_names = [ch for chans in ROIS.values() for ch in chans]
    ch_names += [ch for ch in STANDARD_19 if ch not in ch_names]
    xyz = sensor_points(scene, ch_names)
    nodes = roi_nodes(scene, ch_names, xyz)
    table = ddtf_rows(tests, question)
    strengths = [v for row in table for v in row[3:5]]
    norm = Normalize(min(strengths), max(strengths))
    pn = p_norm()
    unit = Normalize(0, 1)

    def top(arrows, key):
        return sorted(arrows, key=key)[:DDTF_TOP_EDGES]

    bands = {"Theta": "4–8 Hz", "Alpha": "8–13 Hz"}
    rows = []
    for band in bands:
        sel = [r for r in table if r[0] == band]
        rows += [
            (band, label_a, top([(s, t, c, q) for _, s, t, c, _, q in sel],
                                     lambda a: -a[2]), norm, STRENGTH_CMAP),
            (band, label_b, top([(s, t, a, q) for _, s, t, _, a, q in sel],
                                       lambda a: -a[2]), norm, STRENGTH_CMAP),
            (band, "P values", top([(s, t, pn(q), q) for _, s, t, _, _, q in sel],
                                lambda a: a[3]), unit, P_CMAP),
        ]
    views = list(scene.views)
    ink, muted = "#1A1A1A", "#4D4D4D"

    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix"}):
        fig_w = 3.6 * len(views) + 3.2
        fig = plt.figure(figsize=(fig_w, 3.6 * len(rows) + 2.6), dpi=300)
        # row labels need a fixed width in inches whatever the number of view columns
        left, right, top_y, bottom_y = 0.075 + 2.7 / fig_w, 0.985, 0.965, 0.115
        cell_w = (right - left) / len(views)
        cell_h = (top_y - bottom_y) / len(rows)

        # shaded panel behind each band's rows, band name running down its left edge
        for b, band in enumerate(bands):
            y1 = top_y - 3 * b * cell_h
            y0 = y1 - 3 * cell_h
            fig.patches.append(FancyBboxPatch(
                (0.025, y0 + 0.004), 0.96, 3 * cell_h - 0.008, transform=fig.transFigure,
                boxstyle="round,pad=0,rounding_size=0.008", facecolor="#F5F6F8",
                edgecolor="#D9DCE1", linewidth=1.0, zorder=-1))
            fig.text(0.045, (y0 + y1) / 2, f"{band}  ({bands[band]})", rotation=90,
                     fontsize=30, fontweight="bold", ha="center", va="center", color=ink)

        for r, (band, label, arrows, nrm, cmap) in enumerate(rows):
            y0 = top_y - (r + 1) * cell_h
            for c, view in enumerate(views):
                ax = fig.add_axes([left + c * cell_w, y0, cell_w, cell_h], projection="3d")
                ax.patch.set_alpha(0)
                draw_roi_network(ax, view, scene, ch_names, xyz, nodes, arrows, nrm, cmap)
            fig.text(0.075, y0 + cell_h / 2, label, fontsize=26, ha="left", va="center",
                     color=ink)
        for c, view in enumerate(views):
            fig.text(left + (c + 0.5) * cell_w, top_y + 0.004, scene.view_titles[view],
                     fontsize=24, fontstyle="italic", ha="center", va="bottom", color=muted)

        for rect, cmap, lo, hi, caption in [
            ([0.27, 0.072, 0.46, 0.009], STRENGTH_CMAP, f"{norm.vmin:.3f}", f"{norm.vmax:.3f}",
             r"Mean dDTF, source $\rightarrow$ target"),
            ([0.27, 0.022, 0.46, 0.009], P_CMAP, f"p = {ALPHA_SIG}", "p ≤ 0.001",
             "P values"),
        ]:
            cax = fig.add_axes(rect)
            cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
            cax.set_xticks([])
            cax.set_yticks([])
            for spine in cax.spines.values():
                spine.set_edgecolor("#888888")
                spine.set_linewidth(0.8)
            cax.text(-0.015, 0.5, lo, transform=cax.transAxes, ha="right", va="center",
                     fontsize=22, color=ink)
            cax.text(1.015, 0.5, hi, transform=cax.transAxes, ha="left", va="center",
                     fontsize=22, color=ink)
            cax.set_title(caption, fontsize=22, pad=7, color=ink)

        # border around the whole figure
        fig.patches.append(Rectangle((0.004, 0.0 - 0.008), 0.992, 0.996, transform=fig.transFigure,
                                     fill=False, edgecolor="#333333", linewidth=2.0))
        fig_dir = WALLAS_FIG_DIR
        ensure_dir(fig_dir)
        for ext in ("png", "svg"):
            fname = os.path.abspath(os.path.join(fig_dir, f"{out_name}.{ext}"))
            fig.savefig(fname, bbox_inches="tight", pad_inches=0.15, facecolor="white")
            print(f"Saved: {fname}")
        plt.close(fig)


# --------------------------------------------------------------------------
# Q2 posterior-alpha topomaps: AI interaction minus no AI interaction, per Wallas step,
# in the two session windows where the notebook's posterior-alpha cluster test was
# significant (bins centred 35-45% and 65-75% of the session, i.e. 30-50% and 60-80%).
# Significance is tested on the posterior ROI (mean of its channels): Wilcoxon signed-rank
# of per-participant AI - No AI differences, FDR across every testable step x window cell
# (cells with >= Q2_TOPO_MIN_N participants having both states).
# --------------------------------------------------------------------------
Q2_TOPO_STEPS = ["Preparation", "Idea generation", "Implementation", "Verification"]
Q2_TOPO_WINDOWS = {"35-45%": "35–45%", "65-75%": "65–75%"}
Q2_TOPO_MARK_STEP = "Implementation"
Q2_TOPO_VLIM = 0.035
Q2_TOPO_ROI = ["P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
Q2_TOPO_MIN_N = 3


def q2_roi_tests(diff):
    """(window, step) -> (n, p, FDR p) for the posterior-ROI AI - No AI difference."""
    from scipy import stats
    roi = diff.assign(roi=diff[Q2_TOPO_ROI].mean(axis=1))
    cells = {}
    for (win, step), g in roi.groupby(["window", "step"]):
        v = g.roi.dropna().to_numpy()
        if len(v) >= Q2_TOPO_MIN_N:
            cells[(win, step)] = (len(v), stats.wilcoxon(v).pvalue)
    q = stats.false_discovery_control([p for _, p in cells.values()])
    return {k: (n, p, qk) for (k, (n, p)), qk in zip(cells.items(), q)}


def plot_q2_alpha_topomaps(diff, save=True):
    """diff: one row per participant x window x step with the AI - No AI difference in
    baseline-corrected relative alpha for each of STANDARD_19 (q2_alpha_pid_diff).
    save=False returns the figure (for inline display) instead of writing files."""
    roi_tests = q2_roi_tests(diff)
    for (win, step), (n, p, q) in sorted(roi_tests.items()):
        print(f"posterior ROI {win} {step}: n = {n}, Wilcoxon p = {p:.4f}, FDR p = {q:.4f}")
    info = mne.create_info(STANDARD_19, 300.0, "eeg")
    info.set_montage("standard_1020")
    cmap = plt.cm.RdBu_r
    ink, muted = "#1A1A1A", "#4D4D4D"

    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix"}):
        n_rows, n_cols = len(Q2_TOPO_WINDOWS), len(Q2_TOPO_STEPS)
        fig_w = 3.6 * n_cols + 3.0
        fig = plt.figure(figsize=(fig_w, 3.9 * n_rows + 2.6), dpi=300)
        left, right, top_y, bottom_y = 0.075 + 2.4 / fig_w, 0.985, 0.92, 0.2
        cell_w = (right - left) / n_cols
        cell_h = (top_y - bottom_y) / n_rows
        for c, step in enumerate(Q2_TOPO_STEPS):
            fig.text(left + (c + 0.5) * cell_w, top_y + 0.012, step, fontsize=24,
                     fontstyle="italic", ha="center", va="bottom", color=muted)
        for r, (win, win_label) in enumerate(Q2_TOPO_WINDOWS.items()):
            y0 = top_y - (r + 1) * cell_h
            fig.patches.append(FancyBboxPatch(
                (0.02, y0 + 0.006), 0.965, cell_h - 0.012, transform=fig.transFigure,
                boxstyle="round,pad=0,rounding_size=0.01", facecolor="#F5F6F8",
                edgecolor="#D9DCE1", linewidth=1.0, zorder=-1))
            fig.text(0.04, y0 + cell_h / 2, f"Session\nprogress\n{win_label}", fontsize=24,
                     ha="left", va="center", color=ink, linespacing=1.3)
            for c, step in enumerate(Q2_TOPO_STEPS):
                ax = fig.add_axes([left + c * cell_w + 0.012, y0 + 0.16 * cell_h,
                                   cell_w - 0.024, cell_h * 0.78])
                g = diff[(diff.window == win) & (diff.step == step)]
                n = len(g)
                if n == 0:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "no participant with\nboth states", ha="center",
                            va="center", fontsize=17, color=muted, transform=ax.transAxes)
                    continue
                values = g[STANDARD_19].mean().to_numpy()
                n, p_roi, q_roi = roi_tests.get((win, step), (n, np.nan, np.nan))
                roi_sig = step == Q2_TOPO_MARK_STEP and q_roi < ALPHA_SIG
                mask = np.isin(STANDARD_19, Q2_TOPO_ROI) if roi_sig else None
                im, cn = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap=cmap,
                                     vlim=(-Q2_TOPO_VLIM, Q2_TOPO_VLIM), contours=6,
                                     extrapolate="head",
                                     sensors=True, mask=mask,
                                     mask_params=dict(marker="o", markersize=9,
                                                      markerfacecolor="black",
                                                      markeredgecolor="white",
                                                      markeredgewidth=1.5))
                # MNE fills a little past the head outline; clip the map to the outline it
                # drew (the longest line on the axes is the head circle)
                outline = max(ax.lines, key=lambda ln: len(ln.get_xdata())).get_xydata()
                centre = outline.mean(axis=0)
                head = plt.Circle(centre, np.linalg.norm(outline - centre, axis=1).mean(),
                                  transform=ax.transData)
                im.set_clip_path(head)
                if cn is not None:
                    cn.set_clip_path(head)
                if roi_sig:
                    ax.text(0.5, 1.0, f"{stars(q_roi)} posterior α, FDR p = {q_roi:.3f}",
                            ha="center", va="bottom", fontsize=16, fontweight="bold",
                            color=ink, transform=ax.transAxes)
                ax.text(0.5, -0.03, f"n = {n}", ha="center", va="top", fontsize=17,
                        color=muted, transform=ax.transAxes)

        cax = fig.add_axes([0.27, 0.105, 0.46, 0.014])
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
        cax.set_xticks([])
        cax.set_yticks([])
        for spine in cax.spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(0.8)
        cax.text(-0.015, 0.5, f"−{Q2_TOPO_VLIM:.3f}", transform=cax.transAxes, ha="right",
                 va="center", fontsize=22, color=ink)
        cax.text(1.015, 0.5, f"+{Q2_TOPO_VLIM:.3f}", transform=cax.transAxes, ha="left",
                 va="center", fontsize=22, color=ink)
        cax.set_title("Relative alpha power, AI interaction − No AI interaction", fontsize=22,
                      pad=7, color=ink)
        # marker key
        key_y = 0.045
        fig.add_artist(plt.Line2D([0.3], [key_y], marker="o", ms=11, mfc="black", mec="white",
                                  mew=1.5, transform=fig.transFigure))
        fig.text(0.312, key_y, f"Posterior ROI ({', '.join(Q2_TOPO_ROI)}), FDR p < .05",
                 fontsize=20, va="center", color=ink)

        fig.patches.append(Rectangle((0.004, 0.004), 0.992, 0.992, transform=fig.transFigure,
                                     fill=False, edgecolor="#333333", linewidth=2.0))
        if not save:
            return fig
        fig_dir = WALLAS_FIG_DIR
        ensure_dir(fig_dir)
        for ext in ("png", "svg"):
            fname = os.path.abspath(os.path.join(fig_dir, f"q2_alpha_topomaps.{ext}"))
            fig.savefig(fname, bbox_inches="tight", pad_inches=0.15, facecolor="white")
            print(f"Saved: {fname}")
        plt.close(fig)


def run(recompute=False, max_subjects=None, network_bands=("Alpha",)):
    ensure_dir(OUTPUT_DIR)
    plt.rcParams.update({"font.family": "Helvetica", "mathtext.fontset": "stixsans"})
    measures = load_measures(recompute, max_subjects)
    ch_names = [str(c) for c in measures["ch_names"]]
    xyz = sensor_xyz(ch_names)
    print(f"Subjects: N = {len(measures['pids'])}")

    plot_band_power(measures, ch_names, xyz)
    for band in network_bands:
        both = np.concatenate([
            edge_values(measures[f"coh_{cond}_{band}"]).mean(0) for cond in COND_LABELS
        ])
        norm = Normalize(both.min(), both.max())
        for cond in COND_LABELS:
            plot_condition_network(measures, ch_names, xyz, band, cond, norm)
        plot_network_comparison(measures, ch_names, xyz, band, norm)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--bands", nargs="+", default=["Alpha"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.recompute, args.max_subjects, tuple(args.bands))
