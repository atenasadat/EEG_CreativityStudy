import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    get_files_with_event_100,
    plot_tfr_bands
)

def extract_epochs_list(file_list, base_path, standard_channels, trigger_id='100'):
    epochs_list = []
    for f_name in file_list:
        try:
            raw = mne.io.read_raw_eeglab(os.path.join(base_path, f_name), preload=True, verbose=False)
            raw.pick(standard_channels)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            if trigger_id in event_id:
                epochs = mne.Epochs(raw, events, event_id[trigger_id], tmin=-5, tmax=5, baseline=(-0.2, 0), preload=True, verbose=False)
                epochs_list.append(epochs)
        except Exception as e:
            print(f"Error processing {f_name}: {e}")
    return epochs_list

def main():
    # Base paths (matching standard workspace configuration)
    base_path = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    base_path_ICA = "/Users/athenasaghi/Desktop/CleanDATA/CLEAN_BASELINES/ICA/"
    
    # Standard EEG channels (excluding earlobes for visual plots)
    standard_chans = [
        'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1', 
        'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'P8', 'T8', 'Pz'
    ]
    
    print("Gathering files...")
    control_files, treatment_files = get_files_with_event_100(base_path)
    
    print("\nExtracting Epochs format for Control...")
    epochs_c_list = extract_epochs_list(control_files, base_path, standard_chans)
    
    print("\nExtracting Epochs format for Treatment...")
    epochs_t_list = extract_epochs_list(treatment_files, base_path, standard_chans)
    
    # Aggregate all individual subject epochs into continuous blocks for Grand Average TFR computing
    print("\nConcatenating spatial epochs...")
    epochs_c_all = mne.concatenate_epochs(epochs_c_list)
    epochs_t_all = mne.concatenate_epochs(epochs_t_list)
    
    print("\nComputing Time-Frequency Representations (Morlet Wavelets)...")
    
    # We call plot_tfr_bands logically strictly for the Morlet Transformation math (isplot=False)
    # Output spans strictly 1.0 Hz to 50.0 Hz in 1 Hz increments.
    tfr_control = plot_tfr_bands(epochs_c_all, freqs_range=(1.0, 50.0, 1.0), isplot=False)
    tfr_treatment = plot_tfr_bands(epochs_t_all, freqs_range=(1.0, 50.0, 1.0), isplot=False)
    
    # Apply standard scientific relative baselining ('logratio' normalizes the 1/f noise spectrum out)
    print("Formatting spatial mapping schemas and baselines...")
    tfr_control.apply_baseline(baseline=(-0.2, 0), mode='logratio')
    tfr_treatment.apply_baseline(baseline=(-0.2, 0), mode='logratio')
    
    output_dir = "timefreq_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dynamically extract the true physiological 98th percentile boundary. 
    # This natively stretches the red-blue colors to fill the map without washing out into the 0-baseline color.
    val_98 = max(
        np.percentile(np.abs(tfr_control.data), 98),
        np.percentile(np.abs(tfr_treatment.data), 98)
    )
    v_lim = (-val_98, val_98)

    print("\nGenerating TFR Plots...")
    
    conditions = {
        "Unassisted": tfr_control,
        "LLM Assisted": tfr_treatment
    }
    
    # --- 1. Plot Region of Interest (ROI) Averages ---
    print("Rendering Regional (ROI) Averages...")
    
    # Define Regions of Interest (ROIs) to strictly match your established pipeline
    rois = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8'],
        'Central': ['C3', 'Cz', 'C4'],
        'Parietal': ['P3', 'Pz', 'P4', 'P7', 'P8'],
        'Occipital': ['O1', 'O2'],
        'Temporal': ['T7', 'T8']
    }
    
    for roi_name, roi_channels in rois.items():
        print(f"  -> {roi_name} ({len(roi_channels)} channels)")
        for cond_name, tfr_data in conditions.items():
            # Combine dynamically extracts and averages strictly the local ROI
            fig = tfr_data.plot(
                picks=roi_channels,
                combine='mean',
                baseline=None, 
                cmap='RdBu_r', 
                vlim=v_lim,
                show=False,
                title=f"{cond_name} - TFR ({roi_name.replace('_', ' ')} ROI)"
            )[0]
            
            ax = fig.axes[0]
            ax.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
            ax.set_ylabel('Frequency (Hz)', fontweight='bold', fontsize=14)
            ax.tick_params(labelsize=12)
            
            safe_cond = cond_name.replace(' ', '_')
            filename = os.path.join(output_dir, f"TFR_{safe_cond}_ROI_{roi_name}.png")
            fig.savefig(filename, dpi=600, bbox_inches='tight')
            plt.close(fig)

    # --- 2. Plot Individual Channels ---
    for ch_name in standard_chans:
        print(f"Rendering Channel: {ch_name}")
        for cond_name, tfr_data in conditions.items():
            # MNE internally renders a Figure array, we grab the immediate 0th plot structure.
            fig = tfr_data.plot(
                picks=[ch_name],
                baseline=None, # Already applied physically to the data array above
                cmap='RdBu_r', # Standard Gold Standard ERSP/TFR Colormap
                vlim=v_lim,
                show=False,
                title=f"{cond_name} - TFR ({ch_name})"
            )[0]
            
            # Extract main subplot and enhance graphic stylings
            ax = fig.axes[0]
            ax.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
            ax.set_ylabel('Frequency (Hz)', fontweight='bold', fontsize=14)
            ax.tick_params(labelsize=12)
            
            # Save file separated correctly (at 600 DPI for high publication quality)
            safe_cond = cond_name.replace(' ', '_')
            filename = os.path.join(output_dir, f"TFR_{safe_cond}_{ch_name}.png")
            fig.savefig(filename, dpi=600, bbox_inches='tight')
            
            # Kill figure cleanly to prevent memory spiking inside mass loops
            plt.close(fig)

    print(f"\nAll TFR images saved efficiently into {output_dir}/")

if __name__ == "__main__":
    main()
