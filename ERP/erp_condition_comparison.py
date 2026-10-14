"""Main-paper result support: fixed Cluster51 ERP follow-up.

This script tests the F3/C3/P3, -450 to -297 ms window per participant,
reports paired condition statistics, and checks sensitivity to trial counts.
"""

import numpy as np
import pandas as pd

import robust_stats as rs
from erp_shared import build_erp_behavior_table


# Pre-aha window used for the band-power markers.
PRE_AHA_WIN = (-0.5, 0.0)

FRONTOCENTRAL = ["F3", "Fz", "F4", "C3", "Cz", "C4"]
CENTROPARIETAL = ["C3", "Cz", "C4", "P3", "Pz", "P4"]
PARIETAL = ["P3", "Pz", "P4", "P7", "P8"]
POSTERIOR = ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"]
RIGHT_ANTERIOR_TEMPORAL = ["F8", "T8"]
RIGHT_OCCIPITAL = ["P4", "P8", "O2"]

# Same ROI x band combo as erp_behavior_correlations.py.
ROI_GROUPS = {
    "Frontal": ["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8"],
    "Central": ["C3", "Cz", "C4"],
    "Temporal": ["T7", "T8"],
    "Parietal": ["P3", "Pz", "P4", "P7", "P8"],
    "Occipital": ["O1", "O2"],
    "Posterior": ["P3", "Pz", "P4", "P7", "P8", "O1", "O2"],
}

FREQ_BANDS = {
    "Theta": dict(freqs=np.arange(4, 9, 1.0), cyc=2.0),
    "Alpha": dict(freqs=np.arange(8, 14, 1.0), cyc=2.0),
    "UpperAlpha": dict(freqs=np.arange(10, 14, 1.0), cyc=2.0),
    "Beta": dict(freqs=np.arange(13, 31, 2.0), cyc=3.0),
    "Gamma": dict(freqs=np.arange(35, 51, 2.0), cyc=4.0),
}

# Literature marker table. All are tested with the same paired condition test.
CONDITION_MARKER_SPECS = {
    # Insight / restructuring ERPs
    "N1P2_EarlyPerceptual": dict(kind="amp", chans=POSTERIOR, win=(0.100, 0.250),
                                 freqs=None, cyc=None,
                                 title="N1/P2 early perceptual differentiation (100-250 ms)"),
    "N270N320_Conflict": dict(kind="amp", chans=FRONTOCENTRAL, win=(0.270, 0.320),
                              freqs=None, cyc=None,
                              title="N270-N320 mismatch/conflict detection"),
    "N380N400_FrontoCentral": dict(kind="amp", chans=FRONTOCENTRAL, win=(0.380, 0.400),
                                   freqs=None, cyc=None,
                                   title="N380-N400 frontal-central insight negativity"),
    "N400_SemanticConstraint": dict(kind="amp", chans=FRONTOCENTRAL, win=(0.300, 0.500),
                                    freqs=None, cyc=None,
                                    title="N400 semantic constraint / remote association"),
    "P300_P3b_Parietal": dict(kind="amp", chans=PARIETAL, win=(0.300, 0.500),
                              freqs=None, cyc=None,
                              title="P300/P3b parietal novel-association formation"),
    "LPC_CentroParietal": dict(kind="amp", chans=CENTROPARIETAL, win=(0.600, 1.500),
                               freqs=None, cyc=None,
                               title="LPC late positive component (600-1500 ms)"),
    "P800_VeryLatePositive": dict(kind="amp", chans=CENTROPARIETAL, win=(0.800, 1.200),
                                  freqs=None, cyc=None,
                                  title="P800 / very late positivity"),
    "Cluster51": dict(kind="amp", chans=["P3", "C3", "F3"], win=(-0.450, -0.297),
                      freqs=None, cyc=None, title="Cluster 51 pre-aha ERP"),

    # Divergent-thinking event-locked ERPs
    "N2_IdeaGeneration": dict(kind="amp", chans=FRONTOCENTRAL, win=(0.200, 0.350),
                              freqs=None, cyc=None,
                              title="N2 idea-generation reduction marker"),
    "P3_IdeaGeneration": dict(kind="amp", chans=PARIETAL, win=(0.300, 0.500),
                              freqs=None, cyc=None,
                              title="P3 idea-generation reduction marker"),

    # Error / response / feedback components. These are aha-trigger proxies unless
    # error/feedback triggers are added to the dataset.
    "ERN_Ne_ResponseProxy": dict(kind="amp", chans=["Fz", "Cz"], win=(0.000, 0.100),
                                 freqs=None, cyc=None,
                                 title="ERN/Ne response-locked proxy (0-100 ms)"),
    "Pe_ResponseProxy": dict(kind="amp", chans=CENTROPARIETAL, win=(0.200, 0.400),
                             freqs=None, cyc=None,
                             title="Pe conscious error-awareness proxy (200-400 ms)"),
    "FRN_FeedbackProxy": dict(kind="amp", chans=["Fz", "Cz"], win=(0.250, 0.300),
                              freqs=None, cyc=None,
                              title="FRN feedback-negativity proxy (250-300 ms)"),
    "FeedbackP300_Proxy": dict(kind="amp", chans=PARIETAL, win=(0.300, 0.500),
                               freqs=None, cyc=None,
                               title="Feedback P300 proxy (300-500 ms)"),

    # Time-frequency markers
    "Gamma40_RightAnteriorTemporal": dict(kind="tfr", chans=RIGHT_ANTERIOR_TEMPORAL,
                                          win=(-0.300, 0.000),
                                          freqs=np.arange(38, 43, 1.0), cyc=4.0,
                                          title="40 Hz gamma burst, right anterior temporal (-300-0 ms)"),
    "AlphaBrainBlink_RightOccipital": dict(kind="tfr", chans=RIGHT_OCCIPITAL,
                                           win=(-1.800, -1.300),
                                           freqs=FREQ_BANDS["Alpha"]["freqs"], cyc=2.0,
                                           title="Right occipital alpha brain blink (~1.5 s before gamma)"),
    "EarlyAlphaERD": dict(kind="tfr", chans=POSTERIOR, win=(0.500, 1.000),
                          freqs=FREQ_BANDS["Alpha"]["freqs"], cyc=2.0,
                          title="Task-related alpha ERD (500-1000 ms)"),
    "UpperAlphaERS_LateIdeation": dict(kind="tfr", chans=POSTERIOR, win=(0.500, 1.500),
                                       freqs=FREQ_BANDS["UpperAlpha"]["freqs"], cyc=2.0,
                                       title="Upper-alpha ERS as ideation proceeds"),
    "PreparatoryOccipitalAlpha_AhaLocked": dict(kind="tfr", chans=RIGHT_OCCIPITAL,
                                                win=(-2.000, -1.500),
                                                freqs=FREQ_BANDS["Alpha"]["freqs"], cyc=2.0,
                                                title="Preparatory occipital alpha, aha-locked proxy"),
    "PreparatoryFrontalTemporalGamma_AhaLocked": dict(kind="tfr",
                                                      chans=["Fp1", "Fp2", "F3", "Fz", "F4", "F7", "F8", "T7", "T8"],
                                                      win=(-2.000, -1.500),
                                                      freqs=FREQ_BANDS["Gamma"]["freqs"], cyc=4.0,
                                                      title="Preparatory frontal-temporal activity, aha-locked proxy"),
}

for roi_name, chans in ROI_GROUPS.items():
    for band_name, band in FREQ_BANDS.items():
        CONDITION_MARKER_SPECS[f"{roi_name}{band_name}"] = dict(
            kind="tfr",
            chans=chans,
            win=PRE_AHA_WIN,
            freqs=band["freqs"],
            cyc=band["cyc"],
            title=f"{roi_name} {band_name}",
        )


# Only run the data-driven Cluster51 ROI follow-up in this script.
# Other biomarker definitions above are kept for reference, but inactive.
ACTIVE_MARKER_SPECS = {
    "Cluster51": CONDITION_MARKER_SPECS["Cluster51"],
}

N_SUBSAMPLE_ITERATIONS = 5000
RANDOM_SEED = 0


LITERATURE_COVERAGE_NOTES = [
    ("N1/P2", "covered", "N1P2_EarlyPerceptual"),
    ("N270-N320", "covered", "N270N320_Conflict"),
    ("N380-N400 frontal-central", "covered", "N380N400_FrontoCentral"),
    ("P300/P3b", "covered", "P300_P3b_Parietal"),
    ("LPC", "covered", "LPC_CentroParietal"),
    ("P800", "covered", "P800_VeryLatePositive"),
    ("40 Hz right anterior temporal gamma", "covered", "Gamma40_RightAnteriorTemporal"),
    ("Right occipital alpha brain blink", "covered", "AlphaBrainBlink_RightOccipital"),
    ("Preparatory pre-problem state", "proxy only", "requires problem-onset trigger for exact test"),
    ("N2/P3 divergent thinking reductions", "covered as event-locked markers", "N2_IdeaGeneration, P3_IdeaGeneration"),
    ("Alpha ERD / upper-alpha ERS", "covered", "EarlyAlphaERD, UpperAlphaERS_LateIdeation"),
    ("N400 remote semantic pairs", "covered as event-locked marker", "N400_SemanticConstraint"),
    ("ERN/Pe/FRN/feedback P300", "proxy only", "requires error/feedback triggers for exact test"),
]


def marker_groups():
    """Active comparisons: Cluster51 only."""
    return {"Cluster51_ROI": ["Cluster51"]}


def print_literature_coverage():
    print("\nLITERATURE BIOMARKER COVERAGE")
    for marker, status, implementation in LITERATURE_COVERAGE_NOTES:
        print(f"  {marker}: {status} -> {implementation}")


def normality_label(check):
    if check["normal_at_alpha"] is None:
        return check["note"]
    return check["note"]


def print_trial_count_summary(trial_counts, included_pids):
    counts = trial_counts[trial_counts["Pid"].isin(included_pids)].copy()
    if counts.empty:
        print("\nTRIAL COUNTS")
        print("  No matched trial counts available.")
        return

    print("\nTRIAL COUNTS USED IN CLUSTER51 TEST")
    for label, col in [("Control", "n_control"), ("Treatment", "n_treatment")]:
        values = counts[col]
        print(
            f"  {label}: mean={values.mean():.2f}, SD={values.std(ddof=1):.2f}, "
            f"median={values.median():.0f}, range={values.min()}-{values.max()}"
        )
    ratio = counts["n_control"].sum() / counts["n_treatment"].sum()
    print(f"  Total control/treatment epoch ratio: {ratio:.2f}x")


def run_cluster51_trial_count_control(trial_long, included_pids,
                                      n_iter=N_SUBSAMPLE_ITERATIONS,
                                      seed=RANDOM_SEED):
    trials = pd.DataFrame(trial_long["Cluster51"])
    if trials.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    subject_values = []
    for pid in included_pids:
        subject_trials = trials[trials["subject"] == pid]
        control = subject_trials.loc[
            subject_trials["condition"] == 0, "amplitude"
        ].dropna().to_numpy()
        treatment = subject_trials.loc[
            subject_trials["condition"] == 1, "amplitude"
        ].dropna().to_numpy()
        n_keep = min(len(control), len(treatment))
        if n_keep:
            subject_values.append((pid, control, treatment, n_keep))

    rows = []
    for _ in range(n_iter):
        control_means = []
        treatment_means = []
        for _, control, treatment, n_keep in subject_values:
            c = rng.choice(control, size=n_keep, replace=False)
            t = treatment
            if len(treatment) > n_keep:
                t = rng.choice(treatment, size=n_keep, replace=False)
            control_means.append(c.mean())
            treatment_means.append(t.mean())

        wilc = rs.exact_wilcoxon(control_means, treatment_means)
        diff = np.asarray(treatment_means) - np.asarray(control_means)
        rows.append({
            "n_subjects": wilc["n"],
            "mean_control": np.mean(control_means),
            "mean_treatment": np.mean(treatment_means),
            "mean_diff": np.mean(diff),
            "wilcoxon_w": wilc["stat"],
            "p": wilc["p"],
            "method": wilc["method"],
        })

    return pd.DataFrame(rows)


def print_subsample_summary(results):
    if results.empty:
        print("\nTRIAL-COUNT CONTROL")
        print("  Could not run subsampling check.")
        return

    p = results["p"].dropna()
    diff = results["mean_diff"].dropna()
    print("\nTRIAL-COUNT CONTROL: CONTROL SUBSAMPLED TO MATCH TREATMENT")
    print(f"  Iterations: {len(results)}")
    print(f"  Subjects per iteration: {int(results['n_subjects'].median())}")
    print(
        f"  Mean diff median={diff.median():+.3e}, "
        f"95% range={diff.quantile(0.025):+.3e} to {diff.quantile(0.975):+.3e}"
    )
    print(
        f"  Wilcoxon p median={p.median():.4f}, "
        f"95% range={p.quantile(0.025):.4f} to {p.quantile(0.975):.4f}"
    )
    print(f"  Percent of subsamples with p<0.05: {(p < 0.05).mean() * 100:.1f}%")


def main():
    print("\nACTIVE CONDITION COMPARISON")
    print("  Cluster51 only: P3/C3/F3, -450 to -297 ms")
    df, trial_long, trial_counts = build_erp_behavior_table(
        include_trials=True,
        marker_specs=ACTIVE_MARKER_SPECS,
    )
    included_pids = set(df["Pid"])
    print_trial_count_summary(trial_counts, included_pids)

    all_rows = []
    for group_name, keys in marker_groups().items():
        rows = []
        for key in keys:
            # Each marker tested separately: this ONE control column vs its ONE
            # treatment column, paired across subjects. No whole-matrix call.
            wilc = rs.exact_wilcoxon(df[f"{key}_Control"], df[f"{key}_Treatment"])
            diff = df[f"{key}_Treatment"] - df[f"{key}_Control"]
            normality = rs.shapiro_normality(diff)
            rows.append({
                "group": group_name,
                "marker": key,
                "title": CONDITION_MARKER_SPECS[key]["title"],
                "n": wilc["n"],
                "mean_control": df[f"{key}_Control"].mean(),
                "mean_treatment": df[f"{key}_Treatment"].mean(),
                "mean_diff": diff.mean(),
                "wilcoxon_w": wilc["stat"],
                "p_unc": wilc["p"],
                "method": wilc["method"],
                "shapiro_w": normality["w"],
                "shapiro_p": normality["p"],
                "normality": normality_label(normality),
            })
        gdf = pd.DataFrame(rows)
        gdf["p_fdr"] = rs.fdr_bh(gdf["p_unc"])   # FDR within this band family
        all_rows.append(gdf)

        print(f"\n{group_name}")
        for r in gdf.sort_values("p_unc").itertuples():
            print(f"  {r.title}: mean diff={r.mean_diff:+.3e}, "
                f"W={r.wilcoxon_w:.1f}, Wilcoxon p={r.p_unc:.4f}, "
                f"FDR p={r.p_fdr:.4f} ({r.method}), "
                f"diff normality Shapiro p={r.shapiro_p:.4f} ({r.normality})")

    subsample_results = run_cluster51_trial_count_control(trial_long, included_pids)
    print_subsample_summary(subsample_results)

    # out = pd.concat(all_rows, ignore_index=True)
    # print("\nFull Table")
    # print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
