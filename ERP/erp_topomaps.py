import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_files_with_event_100,
    extract_evokeds_epochs
)

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
    
    print("Gathering files...")
    control_files, treatment_files = get_files_with_event_100(base_path)
    
    print("\nExtracting Evokeds for Control...")
    task_evokeds_control, _, _, _ = extract_evokeds_epochs(
        control_files, base_path, base_path_ICA, standard_chans
    )
    
    print("\nExtracting Evokeds for Treatment...")
    task_evokeds_treatment, _, _, _ = extract_evokeds_epochs(
        treatment_files, base_path, base_path_ICA, standard_chans
    )
    
    # Compute Grand Averages
    print("\nComputing Grand Averages...")
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
    
    # 4. Generate the 1x3 Plot Grid with constrained spacing
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300, gridspec_kw={'wspace': 0.1})
    
    conditions = ["Unassisted", "LLM Assisted", "Difference"]
    datas = [data_c, data_t, data_d]
    
    for ax, cond_name, d_slice in zip(axes, conditions, datas):
        im, _ = mne.viz.plot_topomap(
            d_slice,
            evoked_c.info,
            axes=ax,
            show=False,
            names=scalp_chans,
            sensors=True,
            res=256,
            vlim=(vmin, vmax),
            cmap='coolwarm'
        )
        ax.set_title(cond_name, fontweight='bold', pad=15, fontsize=16)

    # Attach single colorbar
    fig.colorbar(im, ax=axes, shrink=0.7, label='Amplitude (µV)')
    
    # Save precisely to the requested directory
    output_dir = "erp_topomaps_res"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, "cluster_51_broadband_comparison.png")
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"\nSaved successfully to: {filename}")

if __name__ == "__main__":
    main()
