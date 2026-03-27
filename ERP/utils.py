import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
from typing import List, Tuple, Optional, Any

DEFAULT_BANDS = [
    ('Delta', 1, 4), 
    ('Theta', 4, 8), 
    ('Alpha', 8, 13), 
    ('Beta', 13, 30),
    ('Low Gamma', 30, 45),
    ('High Gamma', 55, 90)
]

def event_process_eeg_files(
    file_list: List[str], 
    base_path: str, 
    base_path_ica: str, 
    standard_channels: List[str], 
    trigger_id: str = '100'
) -> Tuple[List[mne.Evoked], List[mne.Evoked], Optional[mne.Epochs]]:
    """
    Process EEG files and extract evokeds for task and baseline.
    
    Parameters:
    - file_list: List of EEG file names to process
    - base_path: Path to task/treatment EEG files
    - base_path_ica: Path to baseline EEG files
    - standard_channels: List of channel names to keep
    - trigger_id: Trigger event ID to extract (default: '100')
    
    Returns:
    - task_evokeds: List of task evokeds
    - base_evokeds: List of baseline evokeds
    """
    task_evokeds = []
    base_evokeds = []
    epochs_t = None
    
    for f_name in file_list:
        try:
            # 1. Load EEGLAB .set file
            raw = mne.io.read_raw_eeglab(os.path.join(base_path, f_name), preload=True)
            
            # 2. Harmonize Channels
            raw.pick_channels(standard_channels, ordered=True)
            
            # 3. Epoching Task
            events, event_id = mne.events_from_annotations(raw)
            
            if trigger_id in event_id:
                epochs_t = mne.Epochs(raw, events, event_id[trigger_id], tmin=-2, tmax=2, 
                                      baseline=(-0.2, 0), preload=True, verbose=False)
                n_events = len(epochs_t)
                print(f"File: {f_name} | Count of '{trigger_id}' events: {n_events}")
                task_evokeds.append(epochs_t.average())
            else:
                print(f"Trigger '{trigger_id}' not found in {f_name}")
            
            # 4. Processing Baseline File
            sub_id = f_name.split('_')[0]
            base_f = f"Filter100_{sub_id}_baseline_.set"
            base_path_full = os.path.join(base_path_ica, base_f)
            
            if os.path.exists(base_path_full):
                raw_b = mne.io.read_raw_eeglab(base_path_full, preload=True)
                raw_b.pick_channels(standard_channels, ordered=True)
                b_events = mne.make_fixed_length_events(raw_b, duration=1.0)
                epochs_b = mne.Epochs(raw_b, b_events, tmin=-2, tmax=2, 
                                      baseline=(-0.2, 0), preload=True, verbose=False)
                base_evokeds.append(epochs_b.average())
                
        except Exception as e:
            print(f"-----------------------------Error processing {f_name}: {e}")
    
    return task_evokeds, base_evokeds, epochs_t



def plot_tfr_bands(
    epochs: mne.Epochs, 
    bands: Optional[List[Tuple[str, float, float]]] = None, 
    freqs_range: Tuple[float, float, float] = (1, 100, 0.5), 
    isplot: bool = True
) -> mne.time_frequency.EpochsTFR:
    """
    Calculate and visualize Time-Frequency Representation (TFR) for specific frequency bands.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs object.
    bands : list of tuples, optional
        List of (name, fmin, fmax) tuples. Defaults to standard EEG bands.
    freqs_range : tuple, optional
        (start, stop, step) for frequencies.
    isplot : bool, optional
        Whether to generate plots.
        
    Returns:
    --------
    tfr : mne.time_frequency.EpochsTFR
        Time-frequency representation object
    """
    
    if bands is None:
        bands = DEFAULT_BANDS
    
    # Calculate TFR
    freqs = np.arange(*freqs_range)
    n_cycles = freqs / 2.
    tfr = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, 
                     return_itc=False, average=True)
    
    if isplot:
        # Plot each band
        for band_name, fmin, fmax in bands:
            tfr_band = tfr.copy().crop(fmin=fmin, fmax=fmax)
            
            fig, axes = plt.subplots(5, 6, figsize=(16, 10), constrained_layout=True)
            axes = axes.flatten()
            
            times = tfr_band.times * 1000 
            frequencies = tfr_band.freqs
            
            for i, ax in enumerate(axes):
                if i < len(tfr_band.ch_names):
                    data = tfr_band.data[i]
                    im = ax.pcolormesh(times, frequencies, data, 
                                    shading='gouraud', cmap='RdBu_r') 
                    
                    ax.set_title(tfr_band.ch_names[i], fontweight='bold', fontsize=12)
                    ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
                    
                    if i >= 20: ax.set_xlabel('Time (ms)')
                    if i % 5 == 0: ax.set_ylabel('Hz')
                else:
                    ax.axis('off')
            
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02)
            cbar.set_label('Power', rotation=270, labelpad=20)
            
            plt.suptitle(f'ERSP Grid: {band_name} Band ({fmin}-{fmax} Hz)', fontsize=10)
            plt.show()
        
    return tfr


def plot_grand_average_bands(
    ga_task: mne.Evoked, 
    ga_base: mne.Evoked, 
    bands: Optional[List[Tuple[str, float, float]]] = None
) -> None:
    """
    Plot grand average ERP data filtered by frequency bands.
    
    Parameters:
    -----------
    ga_task : mne.Evoked
        Grand average for task condition
    ga_base : mne.Evoked
        Grand average for baseline condition
    bands : list of tuples, optional
        List of (name, fmin, fmax) tuples. Defaults to standard EEG bands.
    """
    if bands is None:
        bands = DEFAULT_BANDS
    
    times = ga_task.times * 1000  # Convert to ms
    
    for name, fmin, fmax in bands:
        ga_task_band = ga_task.copy().filter(l_freq=fmin, h_freq=fmax, fir_design='firwin')
        ga_base_band = ga_base.copy().filter(l_freq=fmin, h_freq=fmax, fir_design='firwin')
        
        data_task = ga_task_band.get_data()
        data_base = ga_base_band.get_data()

    # 3. Create a new figure for each band
    fig, axes = plt.subplots(6, 6, figsize=(10, 7), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(ga_task.ch_names):
            # Plot Task (Blue) and Baseline (Red)
            # Multiplying by 1e6 to convert Volts to Microvolts (µV)
            ax.plot(times, data_task[i] * 1e6, label='Experiment', color='blue', linewidth=1.2)
            ax.plot(times, data_base[i] * 1e6, label='Baseline', color='#d62728', linestyle='--', linewidth=1.2)
            
            ax.set_title(ga_task.ch_names[i], fontweight='bold', fontsize=10)
            ax.axvline(0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.2)
            
            # EEG standard: Negative is often plotted UP. 
            # If you want Positive DOWN, keep ax.invert_yaxis()
            ax.invert_yaxis()
            
            if i >= 20: ax.set_xlabel('Time (ms)')
            if i % 5 == 0: ax.set_ylabel('µV')
        else:
            ax.axis('off')

    plt.suptitle(f'Grand Average: {name} Band ({fmin}-{fmax} Hz)', fontsize=10, fontweight='bold')
    plt.savefig(f'Grand_Average_{name}_Band.png', dpi=300)  # Save figure as PNG
    plt.show()
    

def plot_tfr_bands_topomap(
    tfr_data: mne.time_frequency.EpochsTFR, 
    ga_evoked: mne.Evoked, 
    band_list: Optional[List[Tuple[str, float, float]]] = None, 
    t_start: float = -0.5, 
    t_stop: float = 0.0, 
    cmap: str = 'RdBu_r', 
    vmin: Optional[float] = None, 
    vmax: Optional[float] = None
):
    """
    Plot time-frequency representation across frequency bands as topomaps.
    
    Parameters:
    -----------
    tfr_data : mne.time_frequency.tfr.AverageTFR
        Time-frequency representation data
    ga_evoked : mne.evoked.Evoked
        Grand average evoked data (for channel info and montage)
    band_list : list of tuples, optional
        Frequency bands as (name, fmin, fmax). Default uses standard bands.
    t_start : float
        Start time in seconds (default: -0.5)
    t_stop : float
        Stop time in seconds (default: 0.0)
    cmap : str
        Colormap for topomaps (default: 'RdBu_r')
    vmin : float, optional
        Minimum value for colormap scale. If None, computed from data.
    vmax : float, optional
        Maximum value for colormap scale. If None, computed from data.
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    if band_list is None:
        band_list = DEFAULT_BANDS
    
    # Filter out A1/A2 to avoid overlap crash
    scalp_chans = [c for c in ga_evoked.ch_names if c not in ['A1', 'A2']]
    info_scalp = ga_evoked.copy().pick(scalp_chans).info
    tfr_scalp = tfr_data.copy().pick(scalp_chans)

    # --- Pre-compute topo data for all bands ---
    all_topo = []
    for _, fmin, fmax in band_list:
        tfr_band = tfr_scalp.copy().crop(tmin=t_start, tmax=t_stop, fmin=fmin, fmax=fmax)
        all_topo.append(tfr_band.data.mean(axis=(1, 2)))

    # --- Compute global symmetric scale if not provided ---
    if vmin is None and vmax is None:
        vmin = min(d.min() for d in all_topo)
        vmax = max(d.max() for d in all_topo)
        # abs_max = max(np.abs(d).max() for d in all_topo)
        # vmin, vmax = -abs_max, abs_max

    # --- Plot ---
    fig, axes = plt.subplots(1, len(band_list), figsize=(18, 6), constrained_layout=True)

    for i, ((name, fmin, fmax), topo_data) in enumerate(zip(band_list, all_topo)):
        im, _ = mne.viz.plot_topomap(
            topo_data, 
            info_scalp, 
            axes=axes[i], 
            show=False,
            names=scalp_chans,
            sensors=True,
            cmap=cmap,
            vlim=(vmin, vmax)
        )
        
        for label, pos in zip(scalp_chans, info_scalp.get_montage().get_positions()['ch_pos'].values()):
            axes[i].text(pos[0], pos[1], label, fontsize=8, ha='center', va='center')
        
        axes[i].set_title(f'{name} ({fmin}-{fmax} Hz)\n{int(t_start*1000)} to {int(t_stop*1000)} ms', fontsize=10)
    
    fig.colorbar(im, ax=axes, shrink=0.6, label='Power Change')
    return fig, axes


def plot_tfr_bands_topomap_indiv(
    tfr_data: mne.time_frequency.EpochsTFR, 
    ga_evoked: mne.Evoked, 
    band_list: Optional[List[Tuple[str, float, float]]] = None, 
    t_start: float = -0.5, 
    t_stop: float = 0.0, 
    cmap: str = 'RdBu_r'
):
    """
    Plot time-frequency representation across frequency bands as topomaps,
    each with its own independent colorbar.
    
    Parameters:
    -----------
    tfr_data : mne.time_frequency.tfr.AverageTFR
        Time-frequency representation data
    ga_evoked : mne.evoked.Evoked
        Grand average evoked data (for channel info and montage)
    band_list : list of tuples, optional
        Frequency bands as (name, fmin, fmax). Default uses standard bands.
    t_start : float
        Start time in seconds (default: -0.5)
    t_stop : float
        Stop time in seconds (default: 0.0)
    cmap : str
        Colormap for topomaps (default: 'RdBu_r')
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    if band_list is None:
        band_list = DEFAULT_BANDS
    
    # Filter out A1/A2 to avoid overlap crash
    scalp_chans = [c for c in ga_evoked.ch_names if c not in ['A1', 'A2']]
    info_scalp = ga_evoked.copy().pick(scalp_chans).info
    tfr_scalp = tfr_data.copy().pick(scalp_chans)

    fig, axes = plt.subplots(1, len(band_list), figsize=(18, 6), constrained_layout=True)

    for i, (name, fmin, fmax) in enumerate(band_list):
        tfr_band = tfr_scalp.copy().crop(tmin=t_start, tmax=t_stop, fmin=fmin, fmax=fmax)
        topo_data = tfr_band.data.mean(axis=(1, 2))

        # Each band gets its own vmin/vmax
        vmin, vmax = topo_data.min(), topo_data.max()

        im, _ = mne.viz.plot_topomap(
            topo_data,
            info_scalp,
            axes=axes[i],
            show=False,
            names=scalp_chans,
            sensors=True,
            cmap=cmap,
            vlim=(vmin, vmax)
        )

        for label, pos in zip(scalp_chans, info_scalp.get_montage().get_positions()['ch_pos'].values()):
            axes[i].text(pos[0], pos[1], label, fontsize=8, ha='center', va='center')

        axes[i].set_title(f'{name} ({fmin}-{fmax} Hz)\n{int(t_start*1000)} to {int(t_stop*1000)} ms', fontsize=10)

        # Individual colorbar next to each topomap
        plt.colorbar(im, ax=axes[i], shrink=0.6, label='Power Change')

    return fig, axes

# def plot_erp_topomaps(evoked, time_slices):
#     """
#     Plot ERP topomaps at multiple time points.
    
#     Parameters:
#     -----------
#     evoked : mne.Evoked
#         Evoked data to plot
#     time_slices : list
#         Time points (in seconds) to plot
#     scalp_chans : list, optional
#         Channel names to use. If None, uses all channels from evoked
#     """

#     scalp_chans = [c for c in evoked.ch_names if c not in ['A1', 'A2']]

#     # Get info from evoked data
#     info_scalp = evoked.copy().pick(scalp_chans).info    
#     fig, axes = plt.subplots(1, len(time_slices), figsize=(22, 5), constrained_layout=True)
    
#     for i, t in enumerate(time_slices):
#         # Slice the data at time t and convert to microvolts
#         data_slice = evoked.copy().pick(scalp_chans).crop(t, t).get_data()[:, 0] * 1e6
        
#         mne.viz.plot_topomap(
#             data_slice, 
#             info_scalp, 
#             axes=axes[i], 
#             show=False,
#             names=scalp_chans, 
#             sensors=True,
#             res=256
#         )
        
#         axes[i].set_title(f'{int(t*1000)} ms', fontweight='bold', fontsize=14)
    
#     plt.suptitle('ERP Progression (Event at 0ms)', fontsize=22, fontweight='bold')
#     plt.show()

standard_chans = [
    'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'A1', 'Fp1', 
    'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'A2', 'P8', 'T8', 'Pz'
]
base_path = '/Users/athenasaghi/Desktop/CleanDATA/clean/'

def get_subject_erp(file_list, target_sub_id, trigger_id='100'):
    """
    Finds all sessions for a specific subject, epochs them, 
    and returns an averaged ERP and the time vector.
    """
    subject_epochs = []
    sub_files = [f for f in file_list if f.startswith(f"{target_sub_id}_")]
    time_vector = None
    
    for f_name in sub_files:
        try:
            raw = mne.io.read_raw_eeglab(os.path.join(base_path, f_name), preload=True)
            raw.pick_channels(standard_chans, ordered=True)
            events, event_id = mne.events_from_annotations(raw)
            
            if trigger_id in event_id:
                # Focusing on the immediate Aha! moment window
                epochs = mne.Epochs(raw, events, event_id[trigger_id], tmin=-0.5, tmax=0.5, 
                                    baseline=(-0.2, 0), preload=True, verbose=False)
                subject_epochs.append(epochs)
                if time_vector is None:
                    time_vector = epochs.times
        except Exception as e:
            print(f"Error processing {f_name}: {e}")
            
    if not subject_epochs:
        return None, None
    
    all_sub_epochs = mne.concatenate_epochs(subject_epochs)
    return all_sub_epochs.average(), time_vector

    
def plot_erp_topomaps_shared(
    evoked: mne.Evoked, 
    time_slices: List[float], 
    band: Optional[Tuple[float, float]] = None
):
    """
    Plot ERP topomaps at multiple time points with a shared colorscale across all maps.
    
    Parameters:
    -----------
    evoked : mne.Evoked
        Evoked data to plot
    time_slices : list
        Time points (in seconds) to plot
    band : tuple, optional
        Frequency band as (fmin, fmax) in Hz (e.g. (8, 13) for alpha).
        If None, broadband ERP is plotted.
    """
    scalp_chans = [c for c in evoked.ch_names if c not in ['A1', 'A2']]
    
    # Filter to band if provided
    evoked_plot = evoked.copy().pick(scalp_chans)
    if band is not None:
        fmin, fmax = band
        evoked_plot = evoked_plot.filter(fmin, fmax)
    
    info_scalp = evoked_plot.info

    # Pre-compute all slices to get global min/max
    all_data = [
        evoked_plot.copy().crop(t, t).get_data()[:, 0] * 1e6
        for t in time_slices
    ]
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)

    fig, axes = plt.subplots(1, len(time_slices), figsize=(22, 5), constrained_layout=True)

    band_str = f'{fmin}-{fmax} Hz' if band is not None else 'Broadband'

    for i, (t, data_slice) in enumerate(zip(time_slices, all_data)):
        im, _ = mne.viz.plot_topomap(
            data_slice,
            info_scalp,
            axes=axes[i],
            show=False,
            names=scalp_chans,
            sensors=True,
            res=256,
            vlim=(vmin, vmax)
        )
        axes[i].set_title(f'{int(t*1000)} ms', fontweight='bold', fontsize=14)

    fig.colorbar(im, ax=axes, shrink=0.6, label='Amplitude (µV)')
    plt.suptitle(f'ERP Progression (Event at 0ms) — {band_str}', fontsize=22, fontweight='bold')
    return fig,axes


def plot_erp_topomaps_indiv(
    evoked: mne.Evoked, 
    time_slices: List[float], 
    band: Optional[Tuple[float, float]] = None
):
    """
    Plot ERP topomaps at multiple time points, each with its own colorscale and colorbar.
    
    Parameters:
    -----------
    evoked : mne.Evoked
        Evoked data to plot
    time_slices : list
        Time points (in seconds) to plot
    band : tuple, optional
        Frequency band as (fmin, fmax) in Hz (e.g. (8, 13) for alpha).
        If None, broadband ERP is plotted.
    """
    scalp_chans = [c for c in evoked.ch_names if c not in ['A1', 'A2']]

    # Filter to band if provided
    evoked_plot = evoked.copy().pick(scalp_chans)
    if band is not None:
        fmin, fmax = band
        evoked_plot = evoked_plot.filter(fmin, fmax)

    info_scalp = evoked_plot.info

    fig, axes = plt.subplots(1, len(time_slices), figsize=(22, 5), constrained_layout=True)

    band_str = f'{fmin}-{fmax} Hz' if band is not None else 'Broadband'

    for i, t in enumerate(time_slices):
        data_slice = evoked_plot.copy().crop(t, t).get_data()[:, 0] * 1e6

        im, _ = mne.viz.plot_topomap(
            data_slice,
            info_scalp,
            axes=axes[i],
            show=False,
            names=scalp_chans,
            sensors=True,
            res=256,
            vlim=(data_slice.min(), data_slice.max())
        )
        axes[i].set_title(f'{int(t*1000)} ms', fontweight='bold', fontsize=14)
        plt.colorbar(im, ax=axes[i], shrink=0.6, label='µV')

    plt.suptitle(f'ERP Progression (Event at 0ms) — {band_str}', fontsize=22, fontweight='bold')
    return fig,axes


