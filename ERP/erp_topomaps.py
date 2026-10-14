"""Main-paper figure support: pre-insight ERP cluster visualization.

This script plots the fixed F3/C3/P3, -450 to -297 ms cluster as scalp maps
and channel waveforms for Unassisted vs LLM-assisted insight epochs.
"""

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from erp_shared import get_files_with_trigger
from utils import extract_evokeds_epochs

CLUSTER_CHANS = ['F3', 'C3', 'P3']
WAVEFORM_WINDOW = (-1.0, 1.0)
COND_COLORS = {"Unassisted": "#2E86DE", "LLM Assisted": "#D6453D"}


def channel_waveforms(evokeds, ch, tmin, tmax):
    """Stack one channel across evokeds (µV): returns times, (n_evokeds, n_times)."""
    cropped = [e.copy().pick([ch]).crop(tmin, tmax) for e in evokeds]
    return cropped[0].times, np.vstack([c.data[0] for c in cropped]) * 1e6


def main():
    # Base paths (matching standard workspace configuration)
    base_path = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    base_path_ICA = "/Users/athenasaghi/Desktop/CleanDATA/CLEAN_BASELINES/ICA/"

    # Standard EEG channels
    standard_chans = [
        'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'A1', 'Fp1',
        'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'A2', 'P8', 'T8', 'Pz'
    ]

    # Exclude non-scalp channels for cleaner topomap projection
    scalp_chans = [c for c in standard_chans if c not in ['A1', 'A2']]

    control_files, treatment_files = get_files_with_trigger(base_path, verbose=True)

    task_evokeds_control, _, _, _ = extract_evokeds_epochs(
        control_files, base_path, base_path_ICA, standard_chans
    )

    task_evokeds_treatment, _, _, _ = extract_evokeds_epochs(
        treatment_files, base_path, base_path_ICA, standard_chans
    )

    # Compute Grand Averages
    ga_control = mne.grand_average(task_evokeds_control)
    ga_treatment = mne.grand_average(task_evokeds_treatment)

    # Compute Difference Wave (Control - Treatment)
    ga_diff = mne.combine_evoked([ga_control, ga_treatment], weights=[1, -1])

    # Target Cluster Window
    tmin, tmax = -0.450, -0.297

    print(f"\nAveraging data between {tmin*1000} ms and {tmax*1000} ms...")

    # 1. Crop the data to the cluster window, filter to target channels
    evoked_c = ga_control.copy().pick(scalp_chans).crop(tmin, tmax)
    evoked_t = ga_treatment.copy().pick(scalp_chans).crop(tmin, tmax)
    evoked_d = ga_diff.copy().pick(scalp_chans).crop(tmin, tmax)

    # 2. Average the time window to get a single spatial array per condition (Convert to µV)
    data_c = evoked_c.data.mean(axis=1) * 1e6
    data_t = evoked_t.data.mean(axis=1) * 1e6
    data_d = evoked_d.data.mean(axis=1) * 1e6

    # 3. Find Shared Color Scale (vmin, vmax)
    all_data = [data_c, data_t, data_d]
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)

    print(f"Applying shared color scale: {vmin:.2f} to {vmax:.2f} µV")

    plt.rcParams.update({"font.family": "Times New Roman"})

    # 4. Layout: top row = 4 topomaps + colorbar, bottom row = F3/C3/P3 waveforms
    fig = plt.figure(figsize=(19, 10.5), dpi=300)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.28,
                             left=0.04, right=0.97, top=0.93, bottom=0.07)
    top = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.15)
    bottom = outer[1].subgridspec(1, 3, wspace=0.22)

    conditions = ["Unassisted", "LLM Assisted", "Difference", "Difference\n(cluster channels)"]
    datas = [data_c, data_t, data_d, data_d]
    cluster_mask = np.array([ch in CLUSTER_CHANS for ch in scalp_chans])

    for col, (cond_name, d_slice) in enumerate(zip(conditions, datas)):
        ax = fig.add_subplot(top[0, col])
        is_cluster_panel = col == 3
        im, _ = mne.viz.plot_topomap(
            d_slice,
            evoked_c.info,
            axes=ax,
            show=False,
            names=[ch if ch in CLUSTER_CHANS else "" for ch in scalp_chans]
            if is_cluster_panel else scalp_chans,
            sensors=True,
            mask=cluster_mask if is_cluster_panel else None,
            mask_params=dict(marker='o', markerfacecolor='#F4B6B6',
                             markeredgecolor='white', linewidth=0, markersize=22),
            res=256,
            vlim=(vmin, vmax),
            cmap='coolwarm'
        )
        ax.set_title(cond_name, fontweight='bold', pad=12, fontsize=18)
        if col == 0:
            ax.text(-0.08, 1.12, "A", transform=ax.transAxes, fontsize=22, fontweight='bold')

    # Attach single colorbar
    cax = fig.add_subplot(top[0, 4])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Amplitude (µV)', fontsize=14)

    # 5. ERP waveforms at the cluster channels (mean ± SEM across recordings)
    t0, t1 = WAVEFORM_WINDOW
    wave_axes = []
    for col, ch in enumerate(CLUSTER_CHANS):
        ax = fig.add_subplot(bottom[0, col], sharey=wave_axes[0] if wave_axes else None)
        wave_axes.append(ax)
        ax.axvspan(tmin, tmax, color='0.85', zorder=0,
                   label=f'Cluster window ({tmin*1000:.0f} to {tmax*1000:.0f} ms)')
        for cond_name, evokeds in [("Unassisted", task_evokeds_control),
                                   ("LLM Assisted", task_evokeds_treatment)]:
            times, waves = channel_waveforms(evokeds, ch, t0, t1)
            mean = waves.mean(axis=0)
            sem = waves.std(axis=0, ddof=1) / np.sqrt(waves.shape[0])
            color = COND_COLORS[cond_name]
            ax.plot(times, mean, color=color, linewidth=1.8,
                    label=f'{cond_name} (n = {waves.shape[0]})')
            ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.9)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xlim(t0, t1)
        ax.set_title(ch, fontweight='bold', fontsize=18)
        ax.set_xlabel('Time relative to insight (s)', fontsize=14)
        if col == 0:
            ax.set_ylabel('Amplitude (µV)', fontsize=14)
            ax.text(-0.12, 1.08, "B", transform=ax.transAxes, fontsize=22, fontweight='bold')
        ax.tick_params(labelsize=12)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    wave_axes[-1].legend(loc='upper right', fontsize=11, frameon=False)

    fig.add_artist(Rectangle(
        (0, 0), 1, 1,
        transform=fig.transFigure,
        fill=False,
        edgecolor="black",
        linewidth=0.8,
        clip_on=False,
    ))

    # Save precisely to the requested directory
    output_dir = os.path.join(os.path.dirname(__file__), "erp_topomaps_res")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, "cluster_51_broadband_comparison.png")
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

    print(f"\nSaved successfully to: {filename}")

if __name__ == "__main__":
    main()
