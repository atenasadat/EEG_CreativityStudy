"""Per-person view of the paper's AFTER-aha power result (Athena's TFR script, commit 8c85cba).

The group numbers come from Athena's own module, imported unchanged:
  ERP/erp_tfr_tf_cluster_pre_event.py (repo)
    build_subject_differences -> per-person ROI-averaged Morlet power (dB vs baseline)
    run_band_window_wilcoxon  -> the 4 ROI/band/window Wilcoxon tests + FDR (printed, captured here)
    mean_sem, smooth_line     -> the group power-profile plotting helpers

Checks
  1 implementation: group Wilcoxon (W, p, FDR p) from the per-person window values must equal
    (a) her function's printout in this run and (b) her earlier full run log (13_athena_tfr_run.log),
    and the included people / epoch counts must match that log.
  2 per person and per window: Unassisted (U), LLM-assisted (LLM), U - LLM; how many people go the
    paper's direction; sign test; leave-one-out Wilcoxon (uncorrected and FDR over the 4 windows);
    most influential people; group test without the 2 most extreme people.
  3 do people with very few aha epochs drive the result?

Outputs (outputs/meeting_1013/)
  24_tfr_per_person.npz            per-person ROI TFRs and band time courses (cache; --recompute rebuilds)
  24_tfr_after_aha_per_person.png  group + per-person power profiles
  24_tfr_after_aha_dots.png        per-person U - LLM window values
  24_tfr_checks.json               all numbers

Run (from any folder):
  (see README for environment variables)
    python 24_tfr_after_aha_per_person.py [--recompute]
"""
import argparse
import contextlib
import datetime
import io
import json
import os
import re
import sys
import warnings
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import RESULTS, OUT, ERP_DIR, REFERENCE  # noqa: E402


ERP_DIR = ERP_DIR
sys.path.insert(0, ERP_DIR)
import erp_tfr_tf_cluster_pre_event as A  # noqa: E402  (Athena's module, unchanged)

import numpy as np  # noqa: E402
import scipy.stats as st  # noqa: E402

OUT = RESULTS
NPZ = os.path.join(OUT, "24_tfr_per_person.npz")
FIG_ROWS = os.path.join(OUT, "24_tfr_after_aha_per_person.png")
FIG_DOTS = os.path.join(OUT, "24_tfr_after_aha_dots.png")
JSON_OUT = os.path.join(OUT, "24_tfr_checks.json")
OLD_LOG = os.path.join(REFERENCE, "13_athena_tfr_run.log")

U_COL, L_COL = "#1f6fd1", "#d62728"
INK, MUTE, GRID = "#222222", "#8a8a8a", "#e6e6e6"
SMOOTH = 5  # samples; the window her plot_power_profiles uses (smooth_line default)

# Direction stated in the paper (Main_v06.tex): frontal alpha higher Unassisted, parietal theta higher LLM.
# Sign is for U - LLM.
PAPER_SIGN = {"Alpha": +1, "Theta": -1}


def wkey(row):
    roi, band, window = row[:3]
    return f"{roi}_{band}_{window}"


def bkey(row):
    return f"{row[0]}_{row[1]}"


def wilcox(t, c):
    """Exactly the call in run_band_window_wilcoxon."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.wilcoxon(t, c, zero_method="wilcox", method="auto", nan_policy="omit")


# ----------------------------------------------------------------------------- data
def compute_from_eeg():
    roi_names = A.band_window_roi_names()
    control_by_roi, treatment_by_roi, included, trial_counts, times = A.build_subject_differences(roi_names)
    data = dict(
        pids=np.array(included),
        times=times,
        freqs=A.FREQS,
        n_unassisted=np.array([c for _, c, _ in trial_counts]),
        n_llm=np.array([t for _, _, t in trial_counts]),
        computed_at=np.array(datetime.datetime.now().isoformat(timespec="seconds")),
    )
    for roi in roi_names:
        data[f"{roi}_control_tfr"] = control_by_roi[roi]
        data[f"{roi}_treatment_tfr"] = treatment_by_roi[roi]
    for row in A.ROI_BAND_WINDOWS:
        roi, band, window, fmin, fmax, tmin, tmax = row
        fm = (A.FREQS >= fmin) & (A.FREQS <= fmax)
        data[f"{bkey(row)}_control_tc"] = control_by_roi[roi][:, fm, :].mean(axis=1)
        data[f"{bkey(row)}_treatment_tc"] = treatment_by_roi[roi][:, fm, :].mean(axis=1)
    np.savez_compressed(NPZ, **data)
    print(f"saved {NPZ}")
    return data


def load_data(recompute):
    if recompute or not os.path.exists(NPZ):
        return compute_from_eeg(), "computed from EEG in this run"
    d = dict(np.load(NPZ))
    return d, f"loaded from {os.path.basename(NPZ)} (computed from EEG at {d['computed_at']})"


# ----------------------------------------------------------------------------- check 1
LINE_RE = re.compile(
    r"^(\w+) (\w+) (Post_\w+): .*W=([\d.]+), p=([\deE.+-]+), fdr=([\deE.+-]+), (.+)$")


def parse_band_lines(text):
    out = {}
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if m:
            roi, band, win, w, p, fdr, direction = m.groups()
            out[f"{roi}_{band}_{win}"] = dict(W=float(w), p=p, fdr=fdr, direction=direction.strip())
    return out


def parse_old_log():
    text = open(OLD_LOG).read()
    pids = re.findall(r"processing_subject=\d+/\d+ pid=(P\d+)", text)
    n_incl = re.findall(r"included_subjects=(\d+)", text)
    trial_line = next((ln for ln in text.splitlines() if ln.startswith("trial_counts:")), None)
    return dict(processed=pids, n_included=int(n_incl[-1]) if n_incl else None,
                trial_counts_line=trial_line, band=parse_band_lines(text))


def group_tests(d, keep=None):
    """Window values + Wilcoxon + FDR across the 4 windows, same code as her function."""
    rows = []
    for row in A.ROI_BAND_WINDOWS:
        roi, band, window, fmin, fmax, tmin, tmax = row
        c_all, t_all = d[f"{roi}_control_tfr"], d[f"{roi}_treatment_tfr"]
        fm = (A.FREQS >= fmin) & (A.FREQS <= fmax)
        tm = (d["times"] >= tmin) & (d["times"] <= tmax)
        c = c_all[:, fm, :][:, :, tm].mean(axis=(1, 2))
        t = t_all[:, fm, :][:, :, tm].mean(axis=(1, 2))
        if keep is not None:
            c, t = c[keep], t[keep]
        res = wilcox(t, c)
        rows.append(dict(key=wkey(row), c=c, t=t, W=float(res.statistic), p=float(res.pvalue),
                         mean_diff_llm_minus_u=float(np.nanmean(t - c))))
    fdr = A.fdr_bh([r["p"] for r in rows])
    for r, f in zip(rows, fdr):
        r["fdr"] = float(f)
    return rows


# ----------------------------------------------------------------------------- figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=INK, labelsize=7.5, length=2.5)


def window_title(row, g, k, n):
    roi, band, window, fmin, fmax, tmin, tmax = row
    return (f"{roi} {band.lower()} {fmin}-{fmax} Hz\n"
            f"tested window {tmin:g}-{tmax:g} s\n"
            f"W = {g['W']:g}, p = {g['p']:.3f}, FDR p = {g['fdr']:.3f}\n"
            f"{k} of {n} people in paper's direction")


def plot_rows(d, group, per_window):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pids, times = list(d["pids"]), d["times"]
    n, ncol = len(pids), len(A.ROI_BAND_WINDOWS)
    fig = plt.figure(figsize=(3.55 * ncol + 1.2, 2.5 + 1.08 * n + 1.9))
    gs = fig.add_gridspec(n + 2, ncol, height_ratios=[2.3, 0.42] + [1.0] * n,
                          left=0.085, right=0.985, top=1 - 1.85 / fig.get_figheight(),
                          bottom=0.5 / fig.get_figheight(), hspace=0.16, wspace=0.13)

    def draw_window(ax, row):
        ax.axvspan(row[5], row[6], color="0.55", alpha=0.16, lw=0, zorder=0)
        ax.axhline(0, color=MUTE, ls="--", lw=0.8, zorder=1)
        ax.set_xlim(0, times[-1])
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)

    # top row: group mean +- SEM, as plot_power_profiles (mean_sem, then smooth_line on mean and SEM)
    top_axes = []
    for j, row in enumerate(A.ROI_BAND_WINDOWS):
        ax = fig.add_subplot(gs[0, j], sharey=top_axes[0] if top_axes else None)
        top_axes.append(ax)
        draw_window(ax, row)
        for lab, col in (("control", U_COL), ("treatment", L_COL)):
            m, s = A.mean_sem(d[f"{bkey(row)}_{lab}_tc"])
            m, s = A.smooth_line(m, SMOOTH), A.smooth_line(s, SMOOTH)
            ax.fill_between(times, m - s, m + s, color=col, alpha=0.14, lw=0, zorder=2)
            ax.plot(times, m, color=col, lw=1.6, zorder=3)
        pw = per_window[wkey(row)]
        ax.set_title(window_title(row, group[wkey(row)], pw["k_paper_direction"], pw["N"]),
                     fontsize=8.6, color=INK, loc="left")
        style(ax)
        ax.set_xlabel("Time from aha (s)", fontsize=7.5, color=INK, labelpad=1)
        if j == 0:
            ax.set_ylabel(f"Group mean\n+/- SEM, N = {n}\n\nPower (dB)", fontsize=8.5, color=INK)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)

    # per-person rows, each row its own y scale (shared across the row's 4 panels)
    order = list(range(n))
    for r, i in enumerate(order):
        row_axes = []
        curves = []
        for j, row in enumerate(A.ROI_BAND_WINDOWS):
            ax = fig.add_subplot(gs[r + 2, j], sharex=top_axes[j],
                                 sharey=row_axes[0] if row_axes else None)
            row_axes.append(ax)
            draw_window(ax, row)
            for lab, col in (("control", U_COL), ("treatment", L_COL)):
                y = A.smooth_line(d[f"{bkey(row)}_{lab}_tc"][i], SMOOTH)
                curves.append(y)
                ax.plot(times, y, color=col, lw=1.0, zorder=3)
            pw = per_window[wkey(row)]["people"][i]
            good = np.sign(pw["U_minus_LLM"]) == PAPER_SIGN[row[1]]
            ax.text(0.995, 0.97, f"U - LLM {pw['U_minus_LLM']:+.2f} dB", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color=INK if good else MUTE,
                    fontweight="bold" if good else "normal", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.8))
            style(ax)
            if r < n - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Time from aha (s)", fontsize=7.5, color=INK, labelpad=1)
            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        lo = min(float(np.min(c)) for c in curves)
        hi = max(float(np.max(c)) for c in curves)
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        span = hi - lo if hi > lo else 1.0
        row_axes[0].set_ylim(lo - 0.06 * span, hi + 0.42 * span)  # head room for the label
        row_axes[0].set_ylabel(f"{pids[i]}\nU {int(d['n_unassisted'][i])} / LLM {int(d['n_llm'][i])}",
                               fontsize=8.3, color=INK, rotation=0, ha="right", va="center", labelpad=6)

    handles = [Line2D([], [], color=U_COL, lw=2, label="Unassisted"),
               Line2D([], [], color=L_COL, lw=2, label="LLM-assisted")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.985, 1 - 0.1 / fig.get_figheight()))
    fig.text(0.085, 1 - 0.18 / fig.get_figheight(),
             "After-aha power, Athena's TFR pipeline (Morlet, dB vs -2.5 to -2.0 s baseline), "
             "ROI and band averaged",
             fontsize=11, color=INK, va="top", fontweight="bold")
    fig.text(0.085, 1 - 0.48 / fig.get_figheight(),
             f"Lines smoothed with the same {SMOOTH}-sample moving average as her group plot "
             f"({SMOOTH / (1.0 / np.median(np.diff(times))) * 1000:.0f} ms). Grey band = tested window; "
             "dashed line = 0 dB. Rows: one per participant (label: pid, aha epochs Unassisted / LLM), "
             "own y scale per row.\n"
             "Number in each panel = Unassisted minus LLM-assisted window mean; bold = the paper's "
             "direction (alpha: U > LLM, theta: LLM > U), grey = opposite.",
             fontsize=8.3, color=INK, va="top", linespacing=1.4)
    fig.savefig(FIG_ROWS, dpi=150)
    plt.close(fig)
    print(f"saved {FIG_ROWS}")


def plot_dots(d, group, per_window):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pids = list(d["pids"])
    ncol = len(A.ROI_BAND_WINDOWS)
    fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 5.4))
    for ax, row in zip(axes, A.ROI_BAND_WINDOWS):
        roi, band, window, fmin, fmax, tmin, tmax = row
        pw = per_window[wkey(row)]
        diff = np.array([p["U_minus_LLM"] for p in pw["people"]])
        order = np.argsort(diff)
        y = np.arange(len(diff))
        sign = PAPER_SIGN[band]
        good = np.sign(diff[order]) == sign
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.axvline(0, color=INK, lw=1.0, zorder=2)
        ax.scatter(diff[order][good], y[good], s=40, color=INK, zorder=3, edgecolor="white",
                   linewidth=1.2, label="paper's direction")
        ax.scatter(diff[order][~good], y[~good], s=40, facecolor="white", edgecolor=MUTE,
                   linewidth=1.3, zorder=3, label="opposite")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{pids[i]} ({int(d['n_unassisted'][i])}/{int(d['n_llm'][i])})"
                            for i in order], fontsize=7.8, color=INK)
        lim = np.max(np.abs(diff)) * 1.12
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.8, len(diff) - 0.2 + 2.0)
        g = group[wkey(row)]
        ax.set_title(f"{roi} {band.lower()} {fmin}-{fmax} Hz, {tmin:g}-{tmax:g} s\n"
                     f"W = {g['W']:g}, FDR p = {g['fdr']:.3f}; sign test p = {pw['sign_test_p']:.3f}",
                     fontsize=8.8, color=INK)
        ax.text(0.03, 0.985, f"{pw['k_paper_direction']} of {pw['N']} people in the paper's direction",
                transform=ax.transAxes, fontsize=8.4, color=INK, va="top", fontweight="bold",
                zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.text(0.03, 0.935, f"paper: {'Unassisted' if sign > 0 else 'LLM-assisted'} higher "
                f"({'right' if sign > 0 else 'left'} of 0)", transform=ax.transAxes,
                fontsize=7.8, color=INK, va="top", zorder=5,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_xlabel("Unassisted minus LLM-assisted (dB)\n<- LLM higher    Unassisted higher ->",
                      fontsize=8.2, color=INK)
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    h, lab = axes[-1].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=8.5, handletextpad=0.2,
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("After-aha window power, one dot per person (label: pid, aha epochs Unassisted/LLM)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=1.2)
    fig.savefig(FIG_DOTS, dpi=170)
    plt.close(fig)
    print(f"saved {FIG_DOTS}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="rebuild the npz from the EEG files")
    args = ap.parse_args()

    d, source = load_data(args.recompute)
    pids = [str(p) for p in d["pids"]]
    n = len(pids)
    nU, nL = d["n_unassisted"].astype(int), d["n_llm"].astype(int)
    res = {"data_source": source, "N": n, "pids": pids,
           "aha_epochs": {p: [int(a), int(b)] for p, a, b in zip(pids, nU, nL)},
           "aha_epochs_total": [int(nU.sum()), int(nL.sum())]}

    # --- check 1: her function on these arrays, my re-computation, and the old log
    roi_names = A.band_window_roi_names()
    trial_counts = list(zip(pids, nU, nL))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        A.run_band_window_wilcoxon({r: d[f"{r}_control_tfr"] for r in roi_names},
                                   {r: d[f"{r}_treatment_tfr"] for r in roi_names},
                                   d["times"], pids, trial_counts, A.ALPHA)
    her_now_text = buf.getvalue()
    print("---- Athena's run_band_window_wilcoxon on these arrays ----")
    print(her_now_text.rstrip())
    her_now = parse_band_lines(her_now_text)
    old = parse_old_log()

    rows = group_tests(d)
    group = {r["key"]: r for r in rows}
    cmp = {}
    all_ok = True
    for r in rows:
        k = r["key"]
        mine = dict(W=r["W"], p=f"{r['p']:.4g}", fdr=f"{r['fdr']:.4g}",
                    direction=(f"{A.TREATMENT_LABEL} > {A.CONTROL_LABEL}" if r["mean_diff_llm_minus_u"] > 0
                               else f"{A.CONTROL_LABEL} > {A.TREATMENT_LABEL}"))
        ok_now = her_now.get(k) == mine
        ok_old = old["band"].get(k) == mine
        all_ok &= ok_now and ok_old
        cmp[k] = dict(per_person_recompute=dict(mine, p_exact=r["p"], fdr_exact=r["fdr"]),
                      athena_function_this_run=her_now.get(k), athena_log_13=old["band"].get(k),
                      match_function=ok_now, match_log=ok_old)
    her_trial_line = next((ln for ln in her_now_text.splitlines() if ln.startswith("trial_counts:")), None)
    res["check1_reproduces_athena"] = dict(
        all_match=bool(all_ok),
        per_window=cmp,
        included_matches_log=bool(old["processed"] == pids and old["n_included"] == n),
        trial_counts_line_matches_log=bool(her_trial_line == old["trial_counts_line"]),
        trial_counts_line=her_trial_line,
        note="W exact; p and FDR p compared at the 4 significant digits the log prints.")

    # --- paper text vs script
    p_two = A.fdr_bh([group["Parietal_Theta_Post_0_4"]["p"], group["Frontal_Alpha_Post_0_1"]["p"]])
    res["paper_vs_script"] = dict(
        script_baseline_s=list(A.BASELINE_WINDOW), paper_methods_baseline_s=[-0.2, 0.0],
        script_epoch_s=list(A.EPOCH_WINDOW), paper_methods_epoch_s=[-5.0, 5.0],
        script_n_windows_in_fdr=len(A.ROI_BAND_WINDOWS), paper_text_n_windows_in_fdr=2,
        fdr_if_only_the_2_paper_windows=dict(Parietal_Theta_Post_0_4=float(p_two[0]),
                                             Frontal_Alpha_Post_0_1=float(p_two[1])),
        freqs_used=dict(theta=A.FREQS[(A.FREQS >= 4) & (A.FREQS <= 8)].tolist(),
                        alpha=A.FREQS[(A.FREQS >= 8) & (A.FREQS <= 13)].tolist()))

    # --- check 2: per person, per window
    loo = [group_tests(d, keep=np.delete(np.arange(n), i)) for i in range(n)]
    per_window = {}
    for wi, row in enumerate(A.ROI_BAND_WINDOWS):
        k = wkey(row)
        sign = PAPER_SIGN[row[1]]
        c, t = group[k]["c"], group[k]["t"]
        diff = c - t  # U - LLM
        k_dir = int((np.sign(diff) == sign).sum())
        n_nz = int((diff != 0).sum())
        people = [dict(pid=p, U_dB=float(c[i]), LLM_dB=float(t[i]), U_minus_LLM=float(diff[i]),
                       paper_direction=bool(np.sign(diff[i]) == sign), n_U=int(nU[i]), n_LLM=int(nL[i]))
                  for i, p in enumerate(pids)]
        loo_p = np.array([loo[i][wi]["p"] for i in range(n)])
        loo_fdr = np.array([loo[i][wi]["fdr"] for i in range(n)])
        # people whose removal gives the largest LOO p (exact Wilcoxon p is discrete, so keep all ties)
        infl = [int(i) for i in np.argsort(-loo_p, kind='stable') if loo_p[i] >= loo_p.max() - 1e-12]
        extreme = np.argsort(-np.abs(diff))[:2]
        extreme_dir = np.argsort(-(diff * sign))[:2]  # largest differences in the paper's direction

        def drop_test(idx):
            keep = np.setdiff1d(np.arange(n), idx)
            r = wilcox(t[keep], c[keep])
            dk = diff[keep]
            return dict(dropped=[pids[i] for i in idx], N=int(len(keep)),
                        k_paper_direction=int((np.sign(dk) == sign).sum()),
                        W=float(r.statistic), p=float(r.pvalue),
                        sign_test_p=float(st.binomtest(int((np.sign(dk) == sign).sum()),
                                                       int((dk != 0).sum()), 0.5).pvalue),
                        mean_U_minus_LLM=float(dk.mean()), median_U_minus_LLM=float(np.median(dk)))

        few = np.where(nL < 3)[0]
        per_window[k] = dict(
            roi=row[0], band=row[1], freq_hz=[row[3], row[4]], window_s=[row[5], row[6]],
            paper_direction="Unassisted > LLM-assisted" if sign > 0 else "LLM-assisted > Unassisted",
            N=n, k_paper_direction=k_dir,
            sign_test_p=float(st.binomtest(k_dir, n_nz, 0.5).pvalue),
            W=group[k]["W"], p=group[k]["p"], fdr=group[k]["fdr"],
            mean_U_minus_LLM=float(diff.mean()), median_U_minus_LLM=float(np.median(diff)),
            loo=dict(p_min=float(loo_p.min()), p_max=float(loo_p.max()),
                     p_max_when_dropping=[pids[i] for i in infl],
                     n_drops_p_below_05=int((loo_p < 0.05).sum()),
                     fdr_min=float(loo_fdr.min()), fdr_max=float(loo_fdr.max()),
                     n_drops_fdr_below_05=int((loo_fdr < 0.05).sum()),
                     per_drop_p={pids[i]: float(loo_p[i]) for i in range(n)},
                     per_drop_fdr={pids[i]: float(loo_fdr[i]) for i in range(n)}),
            most_influential=[dict(pid=pids[i], p_without=float(loo_p[i]), U_minus_LLM=float(diff[i]),
                                   n_U=int(nU[i]), n_LLM=int(nL[i])) for i in infl],
            without_2_most_extreme_abs=drop_test(extreme),
            without_2_largest_in_paper_direction=drop_test(extreme_dir),
            without_people_with_lt3_llm_epochs=drop_test(few) if len(few) else None,
            abs_diff_vs_n_llm_spearman=dict(zip(("rho", "p"), map(float, st.spearmanr(np.abs(diff), nL)))),
            people=people)
    # FDR over the 4 windows after each window drops its own 2 most extreme / 2 largest-in-direction people
    for tag in ("without_2_most_extreme_abs", "without_2_largest_in_paper_direction"):
        f = A.fdr_bh([per_window[wkey(r)][tag]["p"] for r in A.ROI_BAND_WINDOWS])
        for r, v in zip(A.ROI_BAND_WINDOWS, f):
            per_window[wkey(r)][tag]["fdr_over_4"] = float(v)
    res["per_window"] = per_window

    with open(JSON_OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved {JSON_OUT}")

    plot_rows(d, group, per_window)
    plot_dots(d, group, per_window)

    # ---- printed summary
    print("\n==== summary ====")
    print(json.dumps({k: v for k, v in res.items() if k != "per_window"}, indent=1))
    for k, pw in per_window.items():
        print(f"\n{k}  ({pw['paper_direction']})")
        print(f"  group: W={pw['W']:g} p={pw['p']:.4f} FDR={pw['fdr']:.4f}  mean U-LLM={pw['mean_U_minus_LLM']:+.3f} dB"
              f"  median={pw['median_U_minus_LLM']:+.3f}")
        print(f"  {pw['k_paper_direction']} of {pw['N']} in paper direction, sign test p={pw['sign_test_p']:.4f}")
        lo = pw["loo"]
        print(f"  LOO p {lo['p_min']:.4f}-{lo['p_max']:.4f} (max when dropping any of {', '.join(lo['p_max_when_dropping'])}), "
              f"{lo['n_drops_p_below_05']}/{pw['N']} drops p<.05; LOO FDR {lo['fdr_min']:.4f}-{lo['fdr_max']:.4f}, "
              f"{lo['n_drops_fdr_below_05']}/{pw['N']} drops FDR<.05")
        print("  most influential: " + ", ".join(
            f"{m['pid']} (p without={m['p_without']:.4f}, U-LLM={m['U_minus_LLM']:+.2f}, n={m['n_U']}/{m['n_LLM']})"
            for m in pw["most_influential"]))
        for tag in ("without_2_most_extreme_abs", "without_2_largest_in_paper_direction",
                    "without_people_with_lt3_llm_epochs"):
            x = pw[tag]
            if x is None:
                continue
            extra = f" FDR4={x['fdr_over_4']:.4f}" if "fdr_over_4" in x else ""
            print(f"  {tag}: drop {x['dropped']} -> N={x['N']} W={x['W']:g} p={x['p']:.4f}{extra}, "
                  f"{x['k_paper_direction']}/{x['N']} in direction, sign p={x['sign_test_p']:.4f}")
        print(f"  |U-LLM| vs n_LLM Spearman rho={pw['abs_diff_vs_n_llm_spearman']['rho']:+.2f} "
              f"p={pw['abs_diff_vs_n_llm_spearman']['p']:.3f}")
        print("  per person (pid U LLM U-LLM nU/nLLM): " + "; ".join(
            f"{p['pid']} {p['U_dB']:+.2f} {p['LLM_dB']:+.2f} {p['U_minus_LLM']:+.2f} {p['n_U']}/{p['n_LLM']}"
            for p in pw["people"]))


if __name__ == "__main__":
    main()
