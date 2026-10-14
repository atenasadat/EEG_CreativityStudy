"""Main-paper result: 126 vs 75 confirmed aha moments, Wilcoxon p = .0017.

This script counts EEG trigger-100 aha markers per participant and condition,
then tests Unassisted vs LLM-assisted paired counts.
"""

import os
import warnings

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/fontconfig-cache")

import mne
import pandas as pd
from scipy.stats import wilcoxon

from erp_shared import (
    BASE_PATH,
    TRIGGER_ID,
    get_files_with_trigger,
    list_eeg_files,
    read_eeg_file,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)


def _pid_sort_key(pid):
    return int(pid.replace("P", ""))


def count_aha_events(base_path=BASE_PATH):
    control_files, treatment_files = get_files_with_trigger(
        base_path,
        trigger_id=TRIGGER_ID,
        verbose=True,
    )
    all_rows = []
    for fname in list_eeg_files(base_path):
        lower = fname.lower()
        if "control" in lower:
            condition = "control"
        elif "treatment" in lower:
            condition = "treatment"
        else:
            continue

        raw = read_eeg_file(fname, base_path=base_path, preload=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        aha_count = (
            int((events[:, 2] == event_id[TRIGGER_ID]).sum())
            if TRIGGER_ID in event_id
            else 0
        )
        all_rows.append({
            "Pid": fname.split("_")[0],
            "condition": condition,
            "file": fname,
            "aha_count": aha_count,
            "has_trigger_100": aha_count > 0,
        })

    rows = []
    for condition, files in [("control", control_files), ("treatment", treatment_files)]:
        for fname in files:
            raw = read_eeg_file(fname, base_path=base_path, preload=False)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            aha_count = int((events[:, 2] == event_id[TRIGGER_ID]).sum())
            rows.append({
                "Pid": fname.split("_")[0],
                "condition": condition,
                "file": fname,
                "aha_count": aha_count,
                "has_trigger_100": True,
            })

    all_file_df = pd.DataFrame(all_rows)
    file_df = pd.DataFrame(rows)
    if file_df.empty:
        return pd.DataFrame(), file_df, all_file_df

    count_df = (
        file_df.pivot_table(
            index="Pid",
            columns="condition",
            values="aha_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    file_count_df = (
        file_df.pivot_table(
            index="Pid",
            columns="condition",
            values="file",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"control": "control_files", "treatment": "treatment_files"})
        .rename_axis(None, axis=1)
    )
    df = pd.merge(count_df, file_count_df, on="Pid", how="left")
    for col in ["control", "treatment", "control_files", "treatment_files"]:
        if col not in df:
            df[col] = 0
    df["total"] = df["control"] + df["treatment"]
    df = df.sort_values("Pid", key=lambda s: s.map(_pid_sort_key)).reset_index(drop=True)
    return df, file_df, all_file_df


def print_file_inventory(all_file_df):
    print("\nALL FILES READ")
    print(
        all_file_df[
            ["Pid", "condition", "file", "aha_count", "has_trigger_100"]
        ].to_string(index=False)
    )
    print("\nALL-FILE SUMMARY")
    print(f"  total files read={len(all_file_df)}")
    print(f"  control files read={int((all_file_df['condition'] == 'control').sum())}")
    print(f"  treatment files read={int((all_file_df['condition'] == 'treatment').sum())}")
    print(f"  files with trigger 100={int(all_file_df['has_trigger_100'].sum())}")
    print(f"  files without trigger 100={int((~all_file_df['has_trigger_100']).sum())}")


def print_count_summary(df, title):
    zero_control = df.loc[df["control"] == 0, "Pid"].tolist()
    zero_treatment = df.loc[df["treatment"] == 0, "Pid"].tolist()
    zero_total = df.loc[df["total"] == 0, "Pid"].tolist()

    print(f"\n{title}")
    cols = ["Pid", "control", "treatment", "total", "control_files", "treatment_files"]
    print(df[cols].to_string(index=False))

    print("\nSUMMARY")
    print(f"  subjects={len(df)}")
    print(
        f"  control total={df['control'].sum()}, "
        f"avg/subj={df['control'].mean():.2f}, "
        f"median/subj={df['control'].median():.1f}"
    )
    print(
        f"  treatment total={df['treatment'].sum()}, "
        f"avg/subj={df['treatment'].mean():.2f}, "
        f"median/subj={df['treatment'].median():.1f}"
    )
    print(
        f"  combined total={df['total'].sum()}, "
        f"avg/subj={df['total'].mean():.2f}, "
        f"median/subj={df['total'].median():.1f}"
    )
    print(f"  zero control aha: {zero_control if zero_control else 'none'}")
    print(f"  zero treatment aha: {zero_treatment if zero_treatment else 'none'}")
    print(f"  zero total aha: {zero_total if zero_total else 'none'}")


def print_wilcoxon(df):
    if len(df) == 0:
        print("\nWILCOXON SIGNED-RANK TEST")
        print("  no subjects available")
        return

    treatment_more = wilcoxon(
        df["treatment"],
        df["control"],
        alternative="greater",
        zero_method="wilcox",
    )
    control_more = wilcoxon(
        df["control"],
        df["treatment"],
        alternative="greater",
        zero_method="wilcox",
    )
    two_sided = wilcoxon(
        df["treatment"],
        df["control"],
        alternative="two-sided",
        zero_method="wilcox",
    )

    print("\nWILCOXON SIGNED-RANK TEST")
    print(
        f"  treatment > control: W={treatment_more.statistic:.4f}, "
        f"p={treatment_more.pvalue:.4f}"
    )
    print(
        f"  control > treatment: W={control_more.statistic:.4f}, "
        f"p={control_more.pvalue:.4f}"
    )
    print(
        f"  two-sided difference: W={two_sided.statistic:.4f}, "
        f"p={two_sided.pvalue:.4f}"
    )


def main():
    df, file_df, all_file_df = count_aha_events()
    print_file_inventory(all_file_df)

    print("\nFILE SUMMARY")
    print(f"  control files with trigger 100={int((file_df['condition'] == 'control').sum())}")
    print(f"  treatment files with trigger 100={int((file_df['condition'] == 'treatment').sum())}")

    matched = df[(df["control_files"] > 0) & (df["treatment_files"] > 0)].copy()
    print_count_summary(matched, "SUBJECTS WITH TRIGGER 100 IN BOTH CONDITIONS")
    print_wilcoxon(matched)


if __name__ == "__main__":
    main()
