# import mne
# from utils import *


# base_path = '/Users/athenasaghi/Desktop/CleanDATA/clean/'
# base_path_ICA = '/Users/athenasaghi/Desktop/CleanDATA/CLEAN_BASELINES/'

# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt




# def plot_spectral_comparison(freqs1, control_psds, freqs2, base_psds, 
#                             channel='Oz', fmin=1.0, fmax=45.0,
#                             label1='Control', label2='Baseline',
#                             title='Spectral Power Comparison'):
    
#     # Professional styling for Journal Paper
#     sns.set_theme(style="ticks", context="paper", font_scale=1.3)
    
#     # Distinguishable pastel-like colors for individual subjects
#     color1_light = "#ff9f9b" # Pastel Red/Pink
#     color2_light = "#a1c9f4" # Pastel Blue
    
#     # Noticeably darker and more saturated colors for the average lines
#     color1_dark = "#e65a5a" # Stronger Red/Pink
#     color2_dark = "#3b82c4" # Stronger Blue
    
#     fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
#     # Plot individual subjects PSDs with low opacity (using light pastels)
#     for psd in control_psds:
#         ax.plot(freqs1, psd, color=color1_light, alpha=0.15, linewidth=0.5)
        
#     for psd in base_psds:
#         ax.plot(freqs2, psd, color=color2_light, alpha=0.15, linewidth=0.5)
        
#     # Plot group means with thick lines (using darker pastels)
#     ax.plot(freqs1, control_psds.mean(axis=0), color=color1_dark, linewidth=3, label=label1)
#     ax.plot(freqs2, base_psds.mean(axis=0), color=color2_dark, linewidth=3, label=label2)
    
#     # Axes and labels formatting
#     ax.set_xlabel('Frequency (Hz)', fontweight='bold')
#     ax.set_ylabel('Power (dB)', fontweight='bold')
#     ax.set_title(title, fontweight='bold', pad=15)
#     ax.set_xlim(fmin, fmax)
    
#     # Clean up aesthetics
#     sns.despine(ax=ax, trim=False)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
#     # Refine legend
#     ax.legend(title='', frameon=True, fancybox=False, edgecolor='black')
    
#     plt.tight_layout()
#     return fig

# def compute_psds_from_epochs(epochs, channel='O1', fmin=1.0, fmax=45.0):
#     """
#     Compute PSD from epochs object.
#     """
#     epochs_ch = epochs.copy().pick_channels([channel])
#     spectrum = epochs_ch.compute_psd(method='welch', fmin=fmin, fmax=fmax, verbose=False)
    
#     # Shape is (n_epochs, n_channels, n_freqs)
#     # We want to average across epochs, get the one channel
#     psd = spectrum.get_data()[:, 0, :]  # Get all epochs, first channel, all freqs
    
#     psd = psd.mean(axis=0)  # Average across epochs -> shape (176,)


#     psd_db = 10 * np.log10(psd)
    
#     return psd_db, spectrum.freqs

# def compute_psds_from_epochs_list(epochs_list, channel=None, fmin=1.0, fmax=45.0):
#     """
#     Compute PSDs from a list of epochs objects (one per subject).
#     """
#     all_psds = []
#     freqs = None
    
#     for epochs in epochs_list:
#         psd_db, f = compute_psds_from_epochs(epochs, channel, fmin, fmax)
#         all_psds.append(psd_db)
#         if freqs is None:
#             freqs = f
    
#     return np.array(all_psds), freqs


# def extract_task_epochs_list(file_list, base_path, standard_chans, trigger_id='100'):
#     """Extract task epochs for each subject."""
#     all_epochs = []
    
#     for f_name in file_list:
#         try:
#             raw = mne.io.read_raw_eeglab(os.path.join(base_path, f_name), preload=True, verbose=False)

#             raw.pick_channels(standard_chans, ordered=True)
#             events, event_id = mne.events_from_annotations(raw, verbose=False)
            
#             if trigger_id in event_id:
#                 epochs = mne.Epochs(raw, events, event_id[trigger_id], 
#                                    tmin=-2, tmax=2, baseline=(-0.2, 0), 
#                                    preload=True, verbose=False)
#                 all_epochs.append(epochs)
#         except Exception as e:
#             print(f"Error processing {f_name}: {e}")
    
#     return all_epochs



# control_files, treatment_files = get_files_with_event_100(base_path)

# standard_chans = [
#     'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'A1', 'Fp1', 
#     'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'A2', 'P8', 'T8', 'Pz'
# ]

# task_evokeds_treatment, base_evokeds_treatment , epochs_t_treatment, base_epochs_list_treatment = extract_evokeds_epochs(
#     treatment_files, base_path, base_path_ICA, standard_chans
# )
# task_evokeds_control, base_evokeds_control , epochs_t_control, base_epochs_list_control = extract_evokeds_epochs(
#     control_files, base_path, base_path_ICA, standard_chans
# )


# channel = 'F4'

# task_epochs_control = extract_task_epochs_list(control_files, base_path, standard_chans)
# task_epochs_treatment = extract_task_epochs_list(treatment_files, base_path, standard_chans)

# control_task_psds, freqs_task_control = compute_psds_from_epochs_list(task_epochs_control, channel=channel, fmin=1.0, fmax=30.0)
# treatment_task_psds, freqs_task_treatment = compute_psds_from_epochs_list(task_epochs_treatment, channel=channel, fmin=1.0, fmax=30.0)

# control_base_psds, freqs_base_control = compute_psds_from_epochs_list(base_epochs_list_control, channel=channel, fmin=1.0, fmax=30.0)
# treatment_base_psds, freqs_base_treatment = compute_psds_from_epochs_list(base_epochs_list_treatment, channel=channel, fmin=1.0, fmax=30.0)

# fig1 = plot_spectral_comparison(freqs_task_control, control_task_psds, freqs_base_control, control_base_psds, 
#                                  channel=channel, fmin=1.0, fmax=30.0,
#                                  label1='Unassisted', label2='Baseline',
#                                  title='')
# fig1.savefig(f'control_{channel}_psd_spectrum.png', dpi=300, bbox_inches='tight')

# fig2 = plot_spectral_comparison(freqs_task_treatment, treatment_task_psds, freqs_base_treatment, treatment_base_psds, 
#                                  channel=channel, fmin=1.0, fmax=30.0,
#                                  label1='LLM-Assisted', label2='Baseline',
#                                  title='')
# fig2.savefig(f'treatment_{channel}_psd_spectrum.png', dpi=300, bbox_inches='tight')


# fig3  = plot_spectral_comparison(freqs_task_control, control_task_psds, freqs_task_treatment, treatment_task_psds, 
#                                  channel=channel, fmin=1.0, fmax=30.0,
#                                  label1='Unassisted', label2='LLM-Assisted',
#                                  title='')
# fig3.savefig(f'conditions_{channel}_psd_spectrum.png', dpi=300, bbox_inches='tight')

# import mne
# from utils import *
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# base_path = '/Users/athenasaghi/Desktop/CleanDATA/clean/'
# base_path_ICA = '/Users/athenasaghi/Desktop/CleanDATA/CLEAN_BASELINES/'


# def plot_spectral_comparison(freqs1, control_psds, freqs2, base_psds, 
#                             channel='All Channels (Avg)', fmin=1.0, fmax=45.0,
#                             label1='Control', label2='Baseline',
#                             title='Spectral Power Comparison'):
    
#     # Professional styling for Journal Paper
#     sns.set_theme(style="ticks", context="paper", font_scale=1.3)
    
#     # Distinguishable pastel-like colors for individual subjects
#     color1_light = "#ff9f9b" # Pastel Red/Pink
#     color2_light = "#a1c9f4" # Pastel Blue
    
#     # Noticeably darker and more saturated colors for the average lines
#     color1_dark = "#e65a5a" # Stronger Red/Pink
#     color2_dark = "#3b82c4" # Stronger Blue
    
#     fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
#     # Plot individual subjects PSDs with low opacity (using light pastels)
#     for psd in control_psds:
#         ax.plot(freqs1, psd, color=color1_light, alpha=0.15, linewidth=0.5)
        
#     for psd in base_psds:
#         ax.plot(freqs2, psd, color=color2_light, alpha=0.15, linewidth=0.5)
        
#     # Plot group means with thick lines (using darker pastels)
#     ax.plot(freqs1, control_psds.mean(axis=0), color=color1_dark, linewidth=3, label=label1)
#     ax.plot(freqs2, base_psds.mean(axis=0), color=color2_dark, linewidth=3, label=label2)
    
#     # Axes and labels formatting
#     ax.set_xlabel('Frequency (Hz)', fontweight='bold')
#     ax.set_ylabel('Relative Power (dB)', fontweight='bold')
#     ax.set_title(title, fontweight='bold', pad=15)
#     ax.set_xlim(fmin, fmax)
    
#     # Clean up aesthetics
#     sns.despine(ax=ax, trim=False)
#     ax.grid(True, linestyle='--', alpha=0.5)
    
#     # Refine legend
#     ax.legend(title='', frameon=True, fancybox=False, edgecolor='black')
    
#     plt.tight_layout()
#     return fig


# def compute_psds_from_epochs_all_channels(epochs, channels, fmin=1.0, fmax=45.0, relative=True):
#     """
#     Compute PSD averaged across multiple channels.
    
#     Parameters:
#         epochs: mne.Epochs object
#         channels: list of channels to average
#         fmin, fmax: frequency range
#         relative: if True, normalize by total power
#     """
#     all_channel_psds = []
#     freqs = None
    
#     for channel in channels:
#         try:
#             epochs_ch = epochs.copy().pick_channels([channel])
#             spectrum = epochs_ch.compute_psd(method='welch', fmin=fmin, fmax=fmax, verbose=False)
            
#             # Average across epochs
#             psd = spectrum.get_data()[:, 0, :].mean(axis=0)
#             all_channel_psds.append(psd)
            
#             if freqs is None:
#                 freqs = spectrum.freqs
#         except:
#             continue
    
#     # Average across all channels
#     psd_avg = np.array(all_channel_psds).mean(axis=0)
    
#     if relative:
#         # Normalize by total power
#         total_power = psd_avg.sum()
#         psd_normalized = psd_avg / total_power
#         psd_db = 10 * np.log10(psd_normalized)
#     else:
#         psd_db = 10 * np.log10(psd_avg)
    
#     return psd_db, freqs


# def compute_psds_from_epochs_list_all_channels(epochs_list, channels, fmin=1.0, fmax=45.0, relative=True):
#     """
#     Compute PSDs from a list of epochs objects, averaged across channels.
#     """
#     all_psds = []
#     freqs = None
    
#     for epochs in epochs_list:
#         psd_db, f = compute_psds_from_epochs_all_channels(epochs, channels, fmin, fmax, relative=relative)
#         all_psds.append(psd_db)
#         if freqs is None:
#             freqs = f
    
#     return np.array(all_psds), freqs


# def extract_task_epochs_list(file_list, base_path, standard_chans, trigger_id='100'):
#     """Extract task epochs for each subject."""
#     all_epochs = []
    
#     for f_name in file_list:
#         try:
#             raw = mne.io.read_raw_eeglab(os.path.join(base_path, f_name), preload=True, verbose=False)
#             raw.pick_channels(standard_chans, ordered=True)
#             events, event_id = mne.events_from_annotations(raw, verbose=False)
            
#             if trigger_id in event_id:
#                 epochs = mne.Epochs(raw, events, event_id[trigger_id], 
#                                    tmin=-2, tmax=2, baseline=(-0.2, 0), 
#                                    preload=True, verbose=False)
#                 all_epochs.append(epochs)
#         except Exception as e:
#             print(f"Error processing {f_name}: {e}")
    
#     return all_epochs


# # Main execution
# control_files, treatment_files = get_files_with_event_100(base_path)

# standard_chans = [
#     'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'A1', 'Fp1', 
#     'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'A2', 'P8', 'T8', 'Pz'
# ]

# # Channels to average (exclude reference)
# channels_to_average = [ch for ch in standard_chans if ch not in ['A1', 'A2']]

# print(f"Averaging across {len(channels_to_average)} channels: {channels_to_average}")

# # Extract all epochs
# print("\nExtracting baseline epochs...")
# task_evokeds_treatment, base_evokeds_treatment, epochs_t_treatment, base_epochs_list_treatment = extract_evokeds_epochs(
#     treatment_files, base_path, base_path_ICA, standard_chans
# )
# task_evokeds_control, base_evokeds_control, epochs_t_control, base_epochs_list_control = extract_evokeds_epochs(
#     control_files, base_path, base_path_ICA, standard_chans
# )

# print("Extracting task epochs...")
# task_epochs_control = extract_task_epochs_list(control_files, base_path, standard_chans)
# task_epochs_treatment = extract_task_epochs_list(treatment_files, base_path, standard_chans)

# # Compute PSDs averaged across all channels
# print("\nComputing PSDs (averaged across all channels)...")
# control_task_psds, freqs_task = compute_psds_from_epochs_list_all_channels(
#     task_epochs_control, channels_to_average, fmin=1.0, fmax=30.0, relative=True
# )
# treatment_task_psds, _ = compute_psds_from_epochs_list_all_channels(
#     task_epochs_treatment, channels_to_average, fmin=1.0, fmax=30.0, relative=True
# )

# control_base_psds, freqs_base = compute_psds_from_epochs_list_all_channels(
#     base_epochs_list_control, channels_to_average, fmin=1.0, fmax=30.0, relative=True
# )
# treatment_base_psds, _ = compute_psds_from_epochs_list_all_channels(
#     base_epochs_list_treatment, channels_to_average, fmin=1.0, fmax=30.0, relative=True
# )

# # Create 3 plots
# print("\nGenerating plots...")

# # Plot 1: Control task vs baseline
# fig1 = plot_spectral_comparison(
#     freqs_task, control_task_psds, 
#     freqs_base, control_base_psds, 
#     channel='All Channels (Avg)', fmin=1.0, fmax=30.0,
#     label1='Unassisted', label2='Baseline',
#     title=''
# )
# fig1.savefig('control_vs_baseline_avg_all_channels.png', dpi=300, bbox_inches='tight')

# # Plot 2: Treatment task vs baseline
# fig2 = plot_spectral_comparison(
#     freqs_task, treatment_task_psds, 
#     freqs_base, treatment_base_psds, 
#     channel='All Channels (Avg)', fmin=1.0, fmax=30.0,
#     label1='LLM-Assisted', label2='Baseline',
#     title=''
# )
# fig2.savefig('treatment_vs_baseline_avg_all_channels.png', dpi=300, bbox_inches='tight')

# # Plot 3: Control vs Treatment (both task conditions)
# fig3 = plot_spectral_comparison(
#     freqs_task, control_task_psds, 
#     freqs_task, treatment_task_psds, 
#     channel='All Channels (Avg)', fmin=1.0, fmax=30.0,
#     label1='Unassisted', label2='LLM-Assisted',
#     title=''
# )
# fig3.savefig('control_vs_treatment_avg_all_channels.png', dpi=300, bbox_inches='tight')
# plt.show()


import mne
from utils import *
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_path = '/Users/athenasaghi/Desktop/CleanDATA/clean/'
base_path_ICA = '/Users/athenasaghi/Desktop/CleanDATA/CLEAN_BASELINES/ICA/'


def compute_psds_from_epochs_all_channels(epochs, channels, fmin=1.0, fmax=45.0, relative=True):
    """
    Compute PSD averaged across multiple channels.
    """
    all_channel_psds = []
    freqs = None
    
    for channel in channels:
        try:
            epochs_ch = epochs.copy().pick([channel])
            spectrum = epochs_ch.compute_psd(method='welch', fmin=fmin, fmax=fmax, verbose=False)
            
            # Average across epochs
            psd = spectrum.get_data()[:, 0, :].mean(axis=0)
            all_channel_psds.append(psd)
            
            if freqs is None:
                freqs = spectrum.freqs
        except:
            continue
    
    # Average across all channels
    psd_avg = np.array(all_channel_psds).mean(axis=0)
    
    if relative:
        # Normalize by total power
        total_power = psd_avg.sum()
        psd_normalized = psd_avg / total_power
        psd_db = 10 * np.log10(psd_normalized)
    else:
        psd_db = 10 * np.log10(psd_avg)
    
    return psd_db, freqs


def compute_psds_from_epochs_list_all_channels(epochs_list, channels, fmin=1.0, fmax=45.0, relative=True):
    """
    Compute PSDs from a list of epochs objects, averaged across channels.
    """
    all_psds = []
    freqs = None
    
    for epochs in epochs_list:
        psd_db, f = compute_psds_from_epochs_all_channels(epochs, channels, fmin, fmax, relative=relative)
        all_psds.append(psd_db)
        if freqs is None:
            freqs = f
    
    return np.array(all_psds), freqs


def extract_task_epochs_list(file_list, base_path, standard_chans, trigger_id='100'):
    """Extract task epochs for each subject."""
    all_epochs = []
    
    for f_name in file_list:
        try:
            raw = mne.io.read_raw_eeglab(os.path.join(base_path, f_name), preload=True, verbose=False)
            raw.pick(standard_chans)
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            
            if trigger_id in event_id:
                epochs = mne.Epochs(raw, events, event_id[trigger_id], 
                                   tmin=-5, tmax=5, baseline=(-0.2, 0), 
                                   preload=True, verbose=False)
                all_epochs.append(epochs)
        except Exception as e:
            print(f"Error processing {f_name}: {e}")
    
    return all_epochs


# Main execution
control_files, treatment_files = get_files_with_event_100(base_path)

standard_chans = [
    'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'A1', 'Fp1', 
    'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'A2', 'P8', 'T8', 'Pz'
]

# Define Regions of Interest (ROIs) to avoid global blurring
rois = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8'],
    'Central': ['C3', 'Cz', 'C4'],
    'Parietal': ['P3', 'Pz', 'P4', 'P7', 'P8'],
    'Occipital': ['O1', 'O2'],
    'Temporal': ['T7', 'T8']
}

# Extract all epochs ONCE to save time
print("\nExtracting baseline epochs...")
task_evokeds_treatment, base_evokeds_treatment, epochs_t_treatment, base_epochs_list_treatment = extract_evokeds_epochs(
    treatment_files, base_path, base_path_ICA, standard_chans
)
task_evokeds_control, base_evokeds_control, epochs_t_control, base_epochs_list_control = extract_evokeds_epochs(
    control_files, base_path, base_path_ICA, standard_chans
)

print("Extracting task epochs...")
task_epochs_control = extract_task_epochs_list(control_files, base_path, standard_chans)
task_epochs_treatment = extract_task_epochs_list(treatment_files, base_path, standard_chans)

# Colors for plotting
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
color_control = "#D25353"
color_treatment = "#5A9CB5"
color_baseline = "#E9B63B"

fmin, fmax = 1.0, 30.0

# 1. Pre-compute all PSDs to find global Y-axis bounds for consistent scaling
roi_psd_data = {}
global_min, global_max = float("inf"), float("-inf")

for roi_name, channels_to_average in rois.items():
    print(f"\nComputing PSDs for ROI: {roi_name}...")
    control_task_psds, freqs_task = compute_psds_from_epochs_list_all_channels(
        task_epochs_control, channels_to_average, fmin=1.0, fmax=30.0, relative=True
    )
    treatment_task_psds, _ = compute_psds_from_epochs_list_all_channels(
        task_epochs_treatment, channels_to_average, fmin=1.0, fmax=30.0, relative=True
    )
    control_base_psds, freqs_base = compute_psds_from_epochs_list_all_channels(
        base_epochs_list_control, channels_to_average, fmin=1.0, fmax=30.0, relative=True
    )
    treatment_base_psds, _ = compute_psds_from_epochs_list_all_channels(
        base_epochs_list_treatment, channels_to_average, fmin=1.0, fmax=30.0, relative=True
    )
    
    roi_psd_data[roi_name] = {
        'control_task': control_task_psds,
        'treatment_task': treatment_task_psds,
        'control_base': control_base_psds,
        'treatment_base': treatment_base_psds,
        'freqs_task': freqs_task,
        'freqs_base': freqs_base
    }
    
    # Calculate bounds
    current_min = min(control_task_psds.min(), treatment_task_psds.min(), 
                      control_base_psds.min(), treatment_base_psds.min())
    current_max = max(control_task_psds.max(), treatment_task_psds.max(), 
                      control_base_psds.max(), treatment_base_psds.max())
    
    if current_min < global_min: global_min = current_min
    if current_max > global_max: global_max = current_max

# Add 5% padding to the upper limit only, force lower limit to -50
y_padding = (global_max - (-50)) * 0.05
y_lim = (-50, global_max + y_padding)

print(f"\nShared Y-axis limits computed: {y_lim[0]:.2f} to {y_lim[1]:.2f} dB")

# 2. Generate plots with shared scale
for roi_name, data in roi_psd_data.items():
    print(f"Generating combined plot for {roi_name}...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=300)
    fig.suptitle(roi_name, fontweight='bold', fontsize=16)
    
    c_psd = data['control_task']
    t_psd = data['treatment_task']
    cb_psd = data['control_base']
    tb_psd = data['treatment_base']
    f_task = data['freqs_task']
    f_base = data['freqs_base']
    
    # Plot 1: Control vs Baseline
    ax = axes[0]
    for psd in c_psd:
        ax.plot(f_task, psd, color=color_control, alpha=0.15, linewidth=0.5)
    for psd in cb_psd:
        ax.plot(f_base, psd, color=color_baseline, alpha=0.15, linewidth=0.5)
    
    ax.plot(f_task, c_psd.mean(axis=0), color=color_control, linewidth=3, label='Unassisted')
    ax.plot(f_base, cb_psd.mean(axis=0), color=color_baseline, linewidth=3, label='Baseline')
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Relative Power (dB)', fontweight='bold', fontsize=14)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax, trim=False)
    
    # Plot 2: Treatment vs Baseline
    ax = axes[1]
    for psd in t_psd:
        ax.plot(f_task, psd, color=color_treatment, alpha=0.15, linewidth=0.5)
    for psd in tb_psd:
        ax.plot(f_base, psd, color=color_baseline, alpha=0.15, linewidth=0.5)
    
    ax.plot(f_task, t_psd.mean(axis=0), color=color_treatment, linewidth=3, label='LLM-Assisted')
    ax.plot(f_base, tb_psd.mean(axis=0), color=color_baseline, linewidth=3, label='Baseline')
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Relative Power (dB)', fontweight='bold', fontsize=14)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax, trim=False)
    
    # Plot 3: Control vs Treatment
    ax = axes[2]
    for psd in c_psd:
        ax.plot(f_task, psd, color=color_control, alpha=0.15, linewidth=0.5)
    for psd in t_psd:
        ax.plot(f_task, psd, color=color_treatment, alpha=0.15, linewidth=0.5)
    
    ax.plot(f_task, c_psd.mean(axis=0), color=color_control, linewidth=3, label='Unassisted')
    ax.plot(f_task, t_psd.mean(axis=0), color=color_treatment, linewidth=3, label='LLM-Assisted')
    
    ax.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Relative Power (dB)', fontweight='bold', fontsize=14)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.5)
    sns.despine(ax=ax, trim=False)
    
    plt.tight_layout()
    fig.savefig(f'spectral_comparison_{roi_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Prevents memory leak over the loop

print("\nAll ROI plots generated and saved successfully!")