import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_multitaper
from typing import List, Tuple, Optional, Any
import scipy.stats
import scipy
from mne.stats import permutation_cluster_1samp_test


standard_chans = [
    'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'A1', 'Fp1', 
    'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'A2', 'P8', 'T8', 'Pz'
]
base_path = '/Users/athenasaghi/Desktop/CleanDATA/clean/'


DEFAULT_BANDS = [
    ('Delta', 1, 4), 
    ('Theta', 4, 8), 
    ('Alpha', 8, 13), 
    ('Beta', 13, 30),
    ('Low Gamma', 30, 45),
    ('High Gamma', 55, 90)
]

def get_files_with_event_100(base_path, trigger_id='100'):
    """Scan directory for .set files containing event trigger_id"""
    control_files = []
    treatment_files = []
    
    for filename in os.listdir(base_path):
        if filename.endswith('.set'):
            filepath = os.path.join(base_path, filename)
            try:
                raw = mne.io.read_raw_eeglab(filepath, preload=False, verbose=False)
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                
                # Check if trigger_id exists in event_id dictionary (as a string key)
                if trigger_id in event_id:
                    if 'control' in filename:
                        control_files.append(filename)
                    elif 'treatment' in filename:
                        treatment_files.append(filename)
                else:
                    print(f"✗ Trigger '{trigger_id}' not in {filename}. Available: {list(event_id.keys())}")
                    
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return sorted(control_files), sorted(treatment_files)

def run_cluster_permutation_test(control_files, treatment_files, get_subject_erp, 
                                  standard_chans, n_permutations=32768, alpha=0.05,
                                  plot_topomap=True, exclude_chans=None, 
                                  use_adjacency=False, adjacency_threshold=None):
    """
    Run a cluster-based permutation test comparing control vs treatment conditions.
    
    Parameters
    ----------
    control_files : list
        List of control (baseline) file paths or names
    treatment_files : list
        List of treatment file paths or names
    get_subject_erp : callable
        Function that takes (file_list, subject_id) and returns (erp_data, time_vector)
    standard_chans : list
        List of channel names in order matching the data
    n_permutations : int, optional
        Number of permutations for the test (default: 32768)
    alpha : float, optional
        Significance threshold (default: 0.05)
    plot_topomap : bool, optional
        Whether to plot topographic maps for significant clusters (default: True)
    exclude_chans : list, optional
        Channels to exclude from topomap plotting (default: ['A1', 'A2'])
    use_adjacency : bool, optional
        Whether to use spatial adjacency constraint (default: False)
    adjacency_threshold : float, optional
        T-statistic threshold for adjacency mode. If None, uses t-distribution
        with alpha=0.05 (default: None)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 't_obs': observed t-values
        - 'clusters': list of cluster indices or masks
        - 'cluster_p_values': p-value for each cluster
        - 'H0': null distribution
        - 'times_vector': time vector
        - 'significant_clusters': list of dicts with significant cluster info
        - 'X': difference data array (subjects, times, channels)
        - 'figures': list of matplotlib figures (if plot_topomap=True)
        - 'cohens_d': Cohen's d effect size map (if use_adjacency=True)
    """
    if exclude_chans is None:
        exclude_chans = ['A1', 'A2']
    
    # Find common subjects
    sub_ids_control = set(f.split('_')[0] for f in control_files)
    sub_ids_treatment = set(f.split('_')[0] for f in treatment_files)
    common_subjects = sorted(list(sub_ids_control.intersection(sub_ids_treatment)))
    print(f"{len(common_subjects)} subjects have both control and treatment sessions "
          f"and will be included in the analysis.")
    
    # Compute differences for each subject
    diff_list = []
    times_vector = None
    erp_control_template = None  # Store for topomap plotting
    
    for sub_id in common_subjects:
        erp_control, t_vec = get_subject_erp(control_files, sub_id)
        erp_treatment, _ = get_subject_erp(treatment_files, sub_id)
        
        if erp_control and erp_treatment:
            # Control (Human Only) - Treatment (AI+Human)
            diff = erp_control.data - erp_treatment.data
            diff_list.append(diff)
            if times_vector is None: 
                times_vector = t_vec
            if erp_control_template is None:
                erp_control_template = erp_control  # Save for info structure
    
    # Convert to 3D array: (n_subjects, n_channels, n_times)
    X = np.array(diff_list)
    # Transpose to MNE format: (subjects, times, channels)
    X = np.transpose(X, (0, 2, 1))
    
    adjacency = None
    info_subset = None
    picks = None
    X_test = X
    cohens_d = None
    
    if use_adjacency:
        picks = [i for i, ch in enumerate(standard_chans) if ch not in exclude_chans]
        info_subset = erp_control_template.copy().pick(
            [standard_chans[i] for i in picks]
        ).info
        
        # Define spatial adjacency
        adjacency, _ = mne.channels.find_ch_adjacency(info_subset, 'eeg')
        X_test = X[:, :, picks]
        n_subjects = X_test.shape[0]
        
        # Calculate Cohen's d
        mean_diff = X_test.mean(axis=0)
        std_diff = X_test.std(axis=0)
        cohens_d = mean_diff / std_diff
        abs_d = np.abs(cohens_d)
        max_idx = np.unravel_index(abs_d.argmax(), abs_d.shape)
        print(f"\nEffect Size Analysis:")
        print(f"  Peak Cohen's d: {cohens_d[max_idx]:.3f}")
        print(f"  At time: {times_vector[max_idx[0]]:.3f}s")
        print(f"  At channel: {standard_chans[picks[max_idx[1]]]}")
        
        if adjacency_threshold is None:
            adjacency_threshold = scipy.stats.t.ppf(1 - alpha, df=n_subjects - 1)
        
        print(f"\nRunning cluster permutation test with spatial adjacency...")
        print(f"  Threshold: {adjacency_threshold:.2f}")
        print(f"  Permutations: {n_permutations}")
    else:
        print(f"\nRunning cluster permutation test with {n_permutations} iterations...")
    
    
    if use_adjacency:
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            X_test, 
            n_permutations=n_permutations,
            adjacency=adjacency,
            out_type='mask',
            threshold=adjacency_threshold
        )
    else:
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            X_test, n_permutations=n_permutations
        )
    
    # Process and display results
    print("\n" + "="*30)
    print("STATISTICAL RESULTS")
    print("="*30)
    
    significant_clusters = []
    significant_found = False
    figures = []
    
    for i, p_val in enumerate(cluster_p_values):
        if p_val < alpha:
            significant_found = True
            
            if use_adjacency:
                # clusters[i] is a 2D boolean mask (time x channels)
                time_inds, chan_inds = np.where(clusters[i])
                
                # Map back to original channel indices
                original_chan_inds = [picks[c] for c in np.unique(chan_inds)]
                chans = [standard_chans[c] for c in original_chan_inds]
                
                # Calculate mean t-value for directionality
                mean_t = t_obs[clusters[i]].mean()
                
            else:
                # Standard mode: clusters[i] is a tuple of (time_inds, chan_inds)
                time_inds, chan_inds = clusters[i]
                chans = [standard_chans[c] for c in np.unique(chan_inds)]
                mean_t = t_obs[time_inds, chan_inds].mean()
            
            # Map indices to real units
            start_t = times_vector[np.unique(time_inds)[0]]
            end_t = times_vector[np.unique(time_inds)[-1]]
            direction = "Human > AI" if mean_t > 0 else "AI > Human"
            
            # Store cluster info
            cluster_info = {
                'cluster_id': i,
                'p_value': p_val,
                'time_start': start_t,
                'time_end': end_t,
                'direction': direction,
                'mean_t': mean_t,
                'channels': chans,
                'time_inds': time_inds,
                'chan_inds': chan_inds
            }
            significant_clusters.append(cluster_info)
            
            # Print results
            print(f"Cluster {i}:")
            print(f"  P-value:    {p_val:.4f}")
            print(f"  Time:       {start_t:.3f}s to {end_t:.3f}s")
            print(f"  Direction:  {direction} (Mean T: {mean_t:.2f})")
            print(f"  Channels:   {', '.join(chans)}")
            print("-" * 20)
            
            # Plot topographic map for this cluster
            if plot_topomap and erp_control_template is not None:
                if use_adjacency:
                    # Calculate mean T-map for the duration of this cluster
                    t_map_cluster = t_obs[np.unique(time_inds), :].mean(axis=0)
                    
                    # Create a spatial mask for channels that were part of the cluster
                    mask = np.zeros(len(picks), dtype=bool)
                    mask[np.unique(chan_inds)] = True
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im, _ = mne.viz.plot_topomap(
                        t_map_cluster, 
                        info_subset, 
                        axes=ax, 
                        contours=0, 
                        mask=mask, 
                        mask_params=dict(marker='o', markerfacecolor='white', markersize=10),
                        names=[standard_chans[p] for p in picks],
                        show=False
                    )
                    plt.colorbar(im, ax=ax, label='T-statistic')
                    
                else:
                    # Standard mode plotting
                    t_map = t_obs[time_inds, :].mean(axis=0)
                    
                    # Exclude reference channels
                    if picks is None:
                        picks = [i for i, ch in enumerate(standard_chans) if ch not in exclude_chans]
                    t_map_no_ref = t_map[picks]
                    
                    if info_subset is None:
                        info_subset = erp_control_template.copy().pick(
                            [standard_chans[i] for i in picks]
                        ).info
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im, _ = mne.viz.plot_topomap(
                        t_map_no_ref, 
                        info_subset, 
                        axes=ax, 
                        contours=0, 
                        names=[standard_chans[i] for i in picks],
                        show=False
                    )
                    plt.colorbar(im, ax=ax, label='T-statistic')
                
                ax.set_title(f"Cluster {i}: {start_t:.3f}-{end_t:.3f}s (p={p_val:.4f})\n"
                           f"{direction}", fontsize=12, fontweight='bold')
                figures.append(fig)
                plt.show()
    
        if p_val <0.1:
            print(f"Note: Cluster {i} has a trend-level p-value of {p_val:.4f}.")
    if not significant_found:
            print(f"No clusters reached the p < {alpha} significance threshold.")
    
    
    return {
        't_obs': t_obs,
        'clusters': clusters,
        'cluster_p_values': cluster_p_values,
        'H0': H0,
        'times_vector': times_vector,
        'significant_clusters': significant_clusters,
        'X': X,
        'n_subjects': len(common_subjects),
        'figures': figures,
        'cohens_d': cohens_d,  # Only populated if use_adjacency=True
        'adjacency': adjacency,  # Only populated if use_adjacency=True
        'picks': picks  # Channel picks used
    }

def extract_evokeds_epochs(
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
    base_epochs_list = []
    epochs_t = None
    
    for f_name in file_list:
        try:
            raw = mne.io.read_raw_eeglab(os.path.join(base_path, f_name), preload=True)
            raw.pick(standard_channels)
            events, event_id = mne.events_from_annotations(raw)
            epochs_t = mne.Epochs(raw, events, event_id[trigger_id], tmin=-5, tmax=5, baseline=(-0.2, 0), preload=True, verbose=False)
            n_events = len(epochs_t)
            print(f"File: {f_name} | Count of '{trigger_id}' events: {n_events}")
            task_evokeds.append(epochs_t.average())

            sub_id = f_name.split('_')[0]
            base_f = f"{sub_id}_baseline_ICA.set"
            base_path_full = os.path.join(base_path_ica, base_f)
            
            if os.path.exists(base_path_full):
                raw_b = mne.io.read_raw_eeglab(base_path_full, preload=True)
                raw_b.pick(standard_channels)
                b_events = mne.make_fixed_length_events(raw_b, duration=1.0)
                epochs_b = mne.Epochs(raw_b, b_events, tmin=-5, tmax=5, baseline=(-0.2, 0), preload=True, verbose=False)
                base_epochs_list.append(epochs_b)
                base_evokeds.append(epochs_b.average())
                
        except Exception as e:
            print(f"-----------------------------Error processing {f_name}: {e}")
    
    return task_evokeds, base_evokeds, epochs_t, base_epochs_list


def plot_tfr_bands(
    epochs: mne.Epochs, 
    bands: Optional[List[Tuple[str, float, float]]] = None, 
    freqs_range: Tuple[float, float, float] = (1, 100, 0.5), 
    isplot: bool = True,
    channels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 150,
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fontsize_title: int = 12,
    fontsize_label: int = 10
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
    channels : list of str, optional
        List of specific channel names to plot. If None, plots all channels.
    figsize : tuple, optional
        Size (width, height) for each individual plot in inches (default: (6, 6) for square)
    dpi : int, optional
        Resolution for the figure (default: 150)
    cmap : str, optional
        Colormap to use (default: 'RdBu_r')
    vmin : float, optional
        Minimum value for colormap. If None, uses data min.
    vmax : float, optional
        Maximum value for colormap. If None, uses data max.
    fontsize_title : int, optional
        Font size for subplot titles (default: 12)
    fontsize_label : int, optional
        Font size for axis labels (default: 10)
        
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
    # Using epochs.compute_tfr() instead of legacy tfr_morlet
    tfr = epochs.compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles, 
                             return_itc=False, average=True, decim=3)
    
    if isplot:
        # Determine which channels to plot
        if channels is not None:
            # Filter to requested channels that exist in the data
            plot_channels = [ch for ch in channels if ch in tfr.ch_names]
            if len(plot_channels) == 0:
                print(f"Warning: None of the requested channels {channels} found in data.")
                print(f"Available channels: {tfr.ch_names}")
                return tfr
            elif len(plot_channels) < len(channels):
                missing = set(channels) - set(plot_channels)
                print(f"Warning: Channels {missing} not found in data. Plotting available channels.")
            
            # Get indices of channels to plot
            ch_indices = [tfr.ch_names.index(ch) for ch in plot_channels]
        else:
            # Plot all channels
            plot_channels = tfr.ch_names
            ch_indices = list(range(len(plot_channels)))
        
        # Plot each band
        for band_name, fmin, fmax in bands:
            tfr_band = tfr.copy().crop(fmin=fmin, fmax=fmax)
            
            times = tfr_band.times * 1000 
            frequencies = tfr_band.freqs
            
            # Compute global vmin/vmax if not provided
            if vmin is None or vmax is None:
                all_data = tfr_band.data[ch_indices]
                data_vmin = all_data.min() if vmin is None else vmin
                data_vmax = all_data.max() if vmax is None else vmax
            else:
                data_vmin, data_vmax = vmin, vmax
            
            # Create a separate figure for each channel
            for ch_idx, ch_name in zip(ch_indices, plot_channels):
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                
                data = tfr_band.data[ch_idx]
                
                im = ax.pcolormesh(
                    times, frequencies, data, 
                    shading='gouraud', 
                    cmap=cmap,
                    vmin=data_vmin,
                    vmax=data_vmax
                ) 
                
                # Title with band and channel info (no bold)
                ax.set_title(f'{band_name} Band ({fmin}-{fmax} Hz) - Channel {ch_name}', 
                           fontsize=fontsize_title, pad=15)
                
                # Event onset line
                ax.axvline(0, color='white', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
                
                # Set axis labels (no bold)
                ax.set_xlabel('Time (ms)', fontsize=fontsize_label)
                ax.set_ylabel('Frequency (Hz)', fontsize=fontsize_label)
                
                # Improve tick appearance
                ax.tick_params(labelsize=fontsize_label - 1)
                ax.grid(False)
                
                # Set spine width for cleaner look
                for spine in ax.spines.values():
                    spine.set_linewidth(1.0)
                
                # Make plot square by setting aspect ratio
                ax.set_aspect('auto')
                
                # Add colorbar (no bold label)
                cbar = plt.colorbar(im, ax=ax, shrink=0.9, aspect=20)
                cbar.set_label('Power', rotation=270, labelpad=20, 
                              fontsize=fontsize_label)
                cbar.ax.tick_params(labelsize=fontsize_label - 1)
                
                plt.tight_layout()
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
            raw.pick(standard_chans)
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

def extract_erp_measures(evoked, time_window, channels, measure='mean'):
    """
    Extract ERP component measures.
    
    Parameters:
    -----------
    evoked : mne.Evoked
        Evoked data
    time_window : tuple
        (tmin, tmax) in seconds
    channels : list or str
        Channel(s) to analyze
    measure : str
        'mean', 'peak', or 'peak_latency'
    
    Returns:
    --------
    float or dict
        Extracted measure(s)
    """
    # Pick channels and crop time
    data = evoked.copy().pick(channels).crop(*time_window)
    
    if measure == 'mean':
        return data.get_data().mean() * 1e6  # Convert to µV
    elif measure == 'peak':
        return data.get_data().max() * 1e6  # or .min() for negative peaks
    elif measure == 'peak_latency':
        peak_idx = np.argmax(np.abs(data.get_data()))
        return data.times[peak_idx]
    


# Define ROIs
rois = {
    'frontal': ['F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4'],
    'central': ['C3', 'Cz', 'C4'],
    'parietal': ['P3', 'Pz', 'P4', 'CP3', 'CPz', 'CP4']
}

def extract_roi_erp(evoked, roi_channels, time_window):
    """Average ERP across ROI channels"""
    roi_evoked = evoked.copy().pick(roi_channels)
    roi_data = roi_evoked.crop(*time_window).get_data().mean(axis=0)
    return roi_data.mean() * 1e6  # Mean amplitude in µV



def extract_single_trial_amplitudes(epochs, time_window, channel):
    """
    Extract amplitude for each trial.
    
    Returns:
    --------
    amplitudes : array
        Mean amplitude per trial
    """
    epochs_cropped = epochs.copy().pick(channel).crop(*time_window)
    # Shape: (n_trials, n_channels, n_times)
    amplitudes = epochs_cropped.get_data().mean(axis=2).squeeze() * 1e6
    return amplitudes
