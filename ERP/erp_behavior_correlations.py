"""Main-paper exploratory result: creativity scores vs EEG change.

This script computes Kendall associations between baseline creativity measures
and LLM-related EEG markers, then saves the brain-behavior scatter figure.
"""

import os
import warnings

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Unknown types found*")
warnings.filterwarnings("ignore", message="The data contains 'boundary' events*")
warnings.filterwarnings("ignore", message="Limited .* annotation\\(s\\).*")
warnings.filterwarnings("ignore", message="Concatenation of Annotations.*")
warnings.filterwarnings("ignore", message="Data file name in EEG.data .*")

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:
    lowess = None

import robust_stats as rs
from erp_condition_comparison import CONDITION_MARKER_SPECS
from erp_shared import BASE_PATH, build_erp_behavior_table, get_trigger_file_groups


AUT_COLS = ["Fluency", "Flexibility", "Elaboration", "Originality"]
AUT_TOTAL_COL = "AUT_Total"
BEHAVIOR_COLS = ["DAT_Score"] + AUT_COLS + [AUT_TOTAL_COL]
AHA_BEHAVIOR_COLS = ["DAT_Score"]

SELECTED_MARKER_KEYS = [
    "PreparatoryFrontalTemporalGamma_AhaLocked",
    "LeftFrontoCentralCluster",
    # "Cluster51",
    # "ERN_Ne_ResponseProxy",
    # "Pe_ResponseProxy",
    # "N380N400_FrontoCentral",
    # "N400_SemanticConstraint",
    # "N270N320_Conflict",
    # "N2_IdeaGeneration",
    # "P300_P3b_Parietal",
    # "P3_IdeaGeneration",
    # "FeedbackP300_Proxy",
]
LOCAL_MARKER_SPECS = dict(CONDITION_MARKER_SPECS)
LOCAL_MARKER_SPECS["LeftFrontoCentralCluster"] = dict(
    kind="amp",
    chans=["F3", "C3"],
    win=(-0.450, -0.297),
    freqs=None,
    cyc=None,
    title="Left fronto-central pre-aha ERP cluster",
)
BEHAVIOR_MARKER_SPECS = {
    key: LOCAL_MARKER_SPECS[key]
    for key in SELECTED_MARKER_KEYS
}

AHA_COUNT_COLS = ["Aha_Total"]
ALL_MARKER_DIFF_COLS = [f"{key}_Diff" for key in BEHAVIOR_MARKER_SPECS]

ACTIVE_AHA_COMPARISONS = {
    "DAT_Score": ["Aha_Total"],
}

ACTIVE_MARKER_COMPARISONS = {
    "DAT_Score": [
        "PreparatoryFrontalTemporalGamma_AhaLocked_Diff",
    ],
    "Fluency": [
        # "Cluster51_Diff",
        # "ERN_Ne_ResponseProxy_Diff",
        # "Pe_ResponseProxy_Diff",
    ],
    "Flexibility": [
        # "N380N400_FrontoCentral_Diff",
        # "N400_SemanticConstraint_Diff",
        # "Cluster51_Diff",
        # "ERN_Ne_ResponseProxy_Diff",
    ],
    "Elaboration": [
        # "N270N320_Conflict_Diff",
        # "N2_IdeaGeneration_Diff",
    ],
    "Originality": [
        # "P300_P3b_Parietal_Diff",
        # "P3_IdeaGeneration_Diff",
        # "FeedbackP300_Proxy_Diff",
    ],
    AUT_TOTAL_COL: ALL_MARKER_DIFF_COLS,
}

FIGURE_PATH = os.path.join(
    os.path.dirname(__file__),
    "behavior_correlation_stats",
    "brain_behavior_kendall_scatter.png",
)


def add_aha_counts(df, trial_counts):
    counts = trial_counts.copy()
    counts["Aha_Total"] = counts["n_control"] + counts["n_treatment"]
    counts["Aha_Imbalance"] = counts["n_treatment"] - counts["n_control"]
    return pd.merge(df, counts, on="Pid", how="inner")


def add_aut_total_score(df):
    df[AUT_TOTAL_COL] = df[AUT_COLS].sum(axis=1, min_count=len(AUT_COLS))
    return df


def add_erp_shift_scores(df, marker_specs):
    """Add within-subject LLM shift scores: treatment minus control."""
    for key in marker_specs:
        c_col = f"{key}_Control"
        t_col = f"{key}_Treatment"
        df[f"{key}_Diff"] = df[t_col] - df[c_col]
    return df


def print_analysis_files(used_pids, base_path=BASE_PATH):
    """Print the exact trigger-valid EEG files used for included subjects."""
    _, c_groups, t_groups = get_trigger_file_groups(base_path)
    print("\nFILES USED IN ERP-BEHAVIOR CORRELATION ANALYSIS")
    for pid in used_pids:
        print(f"\n{pid}")
        print("  control:")
        for fname in c_groups.get(pid, []):
            print(f"    {fname}")
        print("  treatment:")
        for fname in t_groups.get(pid, []):
            print(f"    {fname}")


def print_normality_checks(df, cols, labels, header):
    print(f"\n{header}")
    print("  Shapiro-Wilk diagnostic only; rank correlations do not require normality.")
    for col in cols:
        check = rs.shapiro_normality(df[col])
        print(
            f"  {labels.get(col, col)}: n={check['n']}, "
            f"W={check['w']:.4f}, p={check['p']:.4f}, {check['note']}"
        )


def rank_correlations(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out = {
        "n": int(x.size),
        "tau": np.nan,
        "tau_p": np.nan,
        "rho": np.nan,
        "rho_p": np.nan,
    }
    if x.size < 3:
        return out
    tau, tau_p = stats.kendalltau(x, y)
    rho, rho_p = stats.spearmanr(x, y)
       
    out.update(
        tau=float(tau),
        tau_p=float(tau_p),
        rho=float(rho),
        rho_p=float(rho_p),

    )
    return out


def print_single_correlations(df, behavior_cols, physiology_cols, labels, header,
                              sample_counts=None, exclude_pairs=None,
                              comparison_map=None, print_all=True):
    sample_counts = sample_counts or {}
    exclude_pairs = exclude_pairs or set()
    print(f"\n{header}")

    rows = []
    for behavior_col in behavior_cols:
        active_cols = comparison_map.get(behavior_col, []) if comparison_map else physiology_cols
        for phys_col in active_cols:
            if (behavior_col, phys_col) in exclude_pairs:
                continue
            corr = rank_correlations(df[behavior_col], df[phys_col])
            rows.append((behavior_col, phys_col, corr))

    if not rows:
        return

    kendall_fdr = np.full(len(rows), np.nan, dtype=float)
    spearman_fdr = np.full(len(rows), np.nan, dtype=float)
    for phys_col in physiology_cols:
        idx = [
            i for i, (_, row_phys_col, _) in enumerate(rows)
            if row_phys_col == phys_col
        ]
        if idx:
            kendall_fdr[idx] = rs.fdr_bh([rows[i][2]["tau_p"] for i in idx])
            spearman_fdr[idx] = rs.fdr_bh([rows[i][2]["rho_p"] for i in idx])

    table_rows = []
    for (behavior_col, phys_col, corr), k_q, s_q in zip(
        rows, kendall_fdr, spearman_fdr
    ):
        row = {
            "behavior": behavior_col,
            "marker": phys_col.replace("_Diff", ""),
            "marker_label": labels.get(phys_col, phys_col),
            "n_subjects": corr["n"],
            "samples": sample_counts.get(phys_col, corr["n"]),
            "tau": corr["tau"],
            "tau_p": corr["tau_p"],
            "tau_p_fdr": k_q,
            "rho": corr["rho"],
            "rho_p": corr["rho_p"],
            "rho_p_fdr": s_q,
   
        }
        table_rows.append(row)

    out = pd.DataFrame(table_rows)
    for r in out.itertuples():

            if r.tau_p < 0.05 or r.rho_p < 0.05:
                print(
                    f"  {r.behavior} vs {r.marker_label}: "
                    f"tau={r.tau:+.4f}, p={r.tau_p:.4f}, p_fdr={r.tau_p_fdr:.4f}, "
                    f"rho={r.rho:+.4f}, rho_p={r.rho_p:.4f}, "
                    f"rho_p_fdr={r.rho_p_fdr:.4f}, "

                    # f"subjects={r.n_subjects}, samples={r.samples}"
                )
    return out


def plot_brain_behavior_results(df, output_path=FIGURE_PATH):
    plot_specs = [
        (
            "DAT_Score",
            "PreparatoryFrontalTemporalGamma_AhaLocked_Diff",
            "DAT score",
            "Prep. frontal-temporal change",
        ),
        (
            AUT_TOTAL_COL,
            "LeftFrontoCentralCluster_Diff",
            "AUT score",
            "Left FC pre-aha ERP change",
        ),
    ]

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=300)
    colors = ["#0057B8", "#D0006F"]

    for ax, (x_col, y_col, x_label, y_label), color in zip(axes, plot_specs, colors):
        x = df[x_col].to_numpy(float)
        y = df[y_col].to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        tau, tau_p = stats.kendalltau(x, y)

        ax.scatter(x, y, s=55, color=color, edgecolors="none", linewidths=0, alpha=0.9)
        if x.size >= 5 and np.unique(x).size > 3:
            order = np.argsort(x)
            x_sorted, y_sorted = x[order], y[order]
            if lowess is not None:
                smooth = lowess(y_sorted, x_sorted, frac=0.8, it=1, return_sorted=True)
                ax.plot(smooth[:, 0], smooth[:, 1], color="black", linewidth=1.8)
            else:
                slope, intercept, _, _ = stats.theilslopes(y, x)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, intercept + slope * x_line, color="black", linewidth=1.8)

        ax.axhline(0, color="0.65", linestyle="--", linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.text(
            0.96, 0.96,
            f"Kendall tau={tau:+.3f}\np={tau_p:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=11,
        )

    fig.tight_layout()
    fig.add_artist(Rectangle(
        (0.005, 0.005), 0.99, 0.99,
        transform=fig.transFigure,
        fill=False,
        edgecolor="black",
        linewidth=0.6,
        clip_on=False,
    ))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved brain-behavior scatter plot: {output_path}")


def main():
    df, _, trial_counts = build_erp_behavior_table(
        include_trials=False,
        marker_specs=BEHAVIOR_MARKER_SPECS,
    )
    df = add_aut_total_score(df)
    df = add_aha_counts(df, trial_counts)
    df = add_erp_shift_scores(df, BEHAVIOR_MARKER_SPECS)
    print_analysis_files(df["Pid"].tolist())

    total_samples = int(df["Aha_Total"].sum())
    control_samples = int(df["n_control"].sum())
    treatment_samples = int(df["n_treatment"].sum())
    print("\nDATA SUMMARY")
    print(f"  subjects={len(df)}")
    print(f"  aha samples total={total_samples}")
    print(f"  control samples={control_samples}")
    print(f"  treatment samples={treatment_samples}")

    aha_cols = AHA_COUNT_COLS
    aha_labels = {
        "Aha_Total": "total aha moments",
    }
    aha_sample_counts = {
        "Aha_Total": total_samples,
    }
    erp_cols = ALL_MARKER_DIFF_COLS
    erp_labels = {
        f"{key}_Diff": f"{spec['title']} LLM shift (treatment - control)"
        for key, spec in BEHAVIOR_MARKER_SPECS.items()
    }
    erp_sample_counts = {col: int(df[col].notna().sum()) for col in erp_cols}

    behavior_labels = {col: col for col in BEHAVIOR_COLS}
    # print_normality_checks(
    #     df, BEHAVIOR_COLS, behavior_labels,
    #     "NORMALITY CHECKS: BEHAVIOR SCORES USED IN CORRELATIONS",
    # )
    # print_normality_checks(
    #     df, aha_cols, aha_labels,
    #     "NORMALITY CHECKS: AHA COUNT VARIABLES USED IN CORRELATIONS",
    # )
    # print_normality_checks(
    #     df, erp_cols, erp_labels,
    #     "NORMALITY CHECKS: ERP SHIFT VARIABLES USED IN CORRELATIONS",
    # )


    print_single_correlations(
        df, AHA_BEHAVIOR_COLS, aha_cols, aha_labels,
        "DAT SCORE x TOTAL AHA MOMENTS",
        sample_counts=aha_sample_counts,
        comparison_map=ACTIVE_AHA_COMPARISONS,
    )
    print_single_correlations(
        df, BEHAVIOR_COLS, erp_cols, erp_labels,
        "CREATIVITY SCORES x ALL CONDITION MARKER LLM EEG SHIFTS",
        sample_counts=erp_sample_counts,
        comparison_map=ACTIVE_MARKER_COMPARISONS,
    )
    plot_brain_behavior_results(df)



if __name__ == "__main__":
    main()
