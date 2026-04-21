import os
import itertools
import numpy as np
import pandas as pd
import mne
import scipy.signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress messy MNE warnings for clean terminal output
warnings.filterwarnings("ignore")

def get_psd_power(raw, channel, fmin, fmax):
    """Calculates specific continuous Welch PSD power for a single target channel."""
    raw_roi = raw.copy().pick([channel])
    spectrum = raw_roi.compute_psd(method='welch', fmin=fmin, fmax=fmax, picks='eeg', verbose=False)
    return np.mean(spectrum.get_data())

def generate_paired_offloading_plot(df, metric_base, t_val, p_val, title_override, out_file, output_dir):
    df_melt = pd.melt(df, id_vars=['Pid'], 
                      value_vars=[f'{metric_base}_Control', f'{metric_base}_Treatment'],
                      var_name='Condition', value_name='Signal Power')
    df_melt['Condition'] = df_melt['Condition'].map({f'{metric_base}_Control': 'Unassisted', f'{metric_base}_Treatment': 'LLM Assisted'})
    
    df_melt = df_melt.dropna(subset=['Signal Power'])
    if df_melt.empty: return
        
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_melt, x='Condition', y='Signal Power', color='grey', boxprops=dict(alpha=0.3))
    sns.swarmplot(data=df_melt, x='Condition', y='Signal Power', color='black', alpha=0.5)
    
    for pid in df['Pid'].unique():
        group = df_melt[df_melt['Pid'] == pid]
        if len(group) == 2:
            p_c = group[group['Condition'] == 'Unassisted']['Signal Power'].values[0]
            p_t = group[group['Condition'] == 'LLM Assisted']['Signal Power'].values[0]
            plt.plot(['Unassisted', 'LLM Assisted'], [p_c, p_t], color='gray', alpha=0.4, linestyle='--')
        
    sig_mark = "***" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")
    plt.title(f"{title_override}\nt-stat: {t_val:.3f} | p-value: {p_val:.4f} ({sig_mark})", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{out_file}.png"), dpi=600)
    plt.close()

def generate_network_topoplot(raw, sig_pairs_dict, band_name, output_dir):
    """Draws red/blue lines between localized functional nodes onto an overhead 2D topographical brain array."""
    if not sig_pairs_dict: return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.axis('off')
    
    # Drop unplottable non-standard reference electrodes to prevent 2D overlap math errors
    safe_raw = raw.copy().drop_channels(['A1', 'A2', 'boundary'], on_missing='ignore')
    
    # Try to extract 2D spatial coordinates for plotting
    try:
        picks = mne.pick_types(safe_raw.info, eeg=True)
        pos = mne.channels.layout._find_topomap_coords(safe_raw.info, picks=picks)
        
        ch_names = [safe_raw.ch_names[p] for p in picks]
        ch_pos = {ch: pos[i] for i, ch in enumerate(ch_names)}
        
        # Inject hyper-realistic academic brain template underneath coordinates using physical extents
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brain_template.png')
        if os.path.exists(template_path):
            img = plt.imread(template_path)
            # Mathematical bounds derived from standard MNE projection geometries
            ax.imshow(img, extent=[-0.125, 0.125, -0.13, 0.13], zorder=0, alpha=0.85)
        else:
            mne.viz.plot_sensors(safe_raw.info, ch_type='eeg', axes=ax, show=False, show_names=False, title="")
            
        # Draw physical electrode nodes directly over the biological image
        for ch, (x, y) in ch_pos.items():
            ax.scatter(x, y, color='black', s=45, zorder=2, edgecolors='white', linewidth=1)
            ax.text(x, y + 0.006, ch, fontsize=8, ha='center', va='bottom', zorder=6, fontweight='bold')
        
    except Exception as e:
        print(f"   [!] Failed to extract brain spatial array for Topoplot. Using placeholder layout. ({e})")
        plt.close()
        return

    ax.set_title(f"Functional Connection Matrix: {band_name}\nRed = Network Hyper-Coupling | Blue = Network Decoupling", fontweight='bold', pad=20)
    
    for (ch1, ch2), t_stat in sig_pairs_dict.items():
        if ch1 not in ch_pos or ch2 not in ch_pos: continue
        
        x1, y1 = ch_pos[ch1]
        x2, y2 = ch_pos[ch2]
        
        # Positive t-stat = Hypercoupled in AI (Red). Negative t-stat = Decoupled in AI (Blue).
        color = 'firebrick' if t_stat > 0 else 'dodgerblue'
        lw = np.clip(abs(t_stat), 1.5, 6) # Dynamic line thickness scaled by mathematical certainty
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.8, zorder=5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Topographical_Network_{band_name}.png"), dpi=600)
    plt.close()

def main():
    # Workspace Configuration
    base_path = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    project_root = "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/"
    output_dir = "continuous_network_dynamics"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("\n=======================================================")
    print("CONTINUOUS NETWORK ASYMMETRY & COHERENCE PIPELINE")
    print("=======================================================")

    # Load Behavioral Trait Scores
    try:
        df_dat = pd.read_csv(os.path.join(project_root, "DATScores.csv"))
        df_aut = pd.read_csv(os.path.join(project_root, "AUTScores.csv"))
        df_dat['Pid'] = df_dat['Pid'].astype(str).str.replace(' ', '')
        df_aut['Pid'] = df_aut['Pid'].astype(str).str.replace(' ', '')
    except FileNotFoundError:
        print("CRITICAL ERROR: Behavioral CSVs not found in project root.")
        return

    # Match Particpants natively ignoring missing insight triggers
    c_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'control' in f.lower()]
    t_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'treatment' in f.lower()]
    
    c_dict = {f.split('_')[0]: f for f in c_files}
    t_dict = {f.split('_')[0]: f for f in t_files}
    matched_pids = sorted(list(set(c_dict.keys()) & set(t_dict.keys())), key=lambda x: int(x.replace('P', '')))
    
    print(f"[SYSTEM] Continuous Extraction Pipeline Booting. N={len(matched_pids)}")
    
    # Pre-extract mathematical channel mapping from the first subject
    samp_raw = mne.io.read_raw_eeglab(os.path.join(base_path, c_dict[matched_pids[0]]), preload=False, verbose=False)
    eeg_chans = [ch for ch in samp_raw.ch_names if ch not in ['A1', 'A2', 'boundary']]
    all_pairs = list(itertools.combinations(eeg_chans, 2))
    print(f"[SYSTEM] Unique Coherence Node Matrix Loaded: {len(all_pairs)} connections per subject.\n")
    
    # Dynamic Frequency Band configuration to cleanly map multiple physiological regions natively
    BANDS = {
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
        'Beta':  (13.0, 30.0),
        'Gamma': (35.0, 50.0)
    }

    results = []
    
    for pid in matched_pids:
        print(f"  -> Processing Holistic Network: {pid}...")
        
        # We only read the gigabytes of data ONE TIME into memory natively per block
        raw_c = mne.io.read_raw_eeglab(os.path.join(base_path, c_dict[pid]), preload=True, verbose=False)
        raw_t = mne.io.read_raw_eeglab(os.path.join(base_path, t_dict[pid]), preload=True, verbose=False)
        
        fs = raw_c.info['sfreq']
        row = {'Pid': pid}
        
        # --- FEATURE 1: Frontal Alpha Asymmetry (FAA) ---
        alpha_f3_c = get_psd_power(raw_c, 'F3', 8, 13)
        alpha_f4_c = get_psd_power(raw_c, 'F4', 8, 13)
        row['FAA_Control'] = np.log(alpha_f4_c) - np.log(alpha_f3_c)
        
        alpha_f3_t = get_psd_power(raw_t, 'F3', 8, 13)
        alpha_f4_t = get_psd_power(raw_t, 'F4', 8, 13)
        row['FAA_Treatment'] = np.log(alpha_f4_t) - np.log(alpha_f3_t)
        
        row['FAA_Shift'] = row['FAA_Treatment'] - row['FAA_Control']
        
        # --- FEATURE 2: Universal Functional Connectivity Brute-Scan ---
        # Highly optimized bypassing MNE metadata layers specifically for speed
        data_c = raw_c.get_data(picks=eeg_chans)
        data_t = raw_t.get_data(picks=eeg_chans)
        
        for ch1, ch2 in all_pairs:
            idx1, idx2 = eeg_chans.index(ch1), eeg_chans.index(ch2)
            
            f_c, cxy_c = scipy.signal.coherence(data_c[idx1], data_c[idx2], fs=fs, nperseg=1024)
            f_t, cxy_t = scipy.signal.coherence(data_t[idx1], data_t[idx2], fs=fs, nperseg=1024)
            
            for band_name, (fmin, fmax) in BANDS.items():
                band_mask = (f_c >= fmin) & (f_c <= fmax)
                row[f'Coh_{band_name}_{ch1}_{ch2}_Control'] = np.mean(cxy_c[band_mask])
                row[f'Coh_{band_name}_{ch1}_{ch2}_Treatment'] = np.mean(cxy_t[band_mask])
            
        results.append(row)
        
    df_eeg = pd.DataFrame(results)
    
    df_final = pd.merge(df_eeg, df_dat[['Pid', 'dat']], on='Pid', how='inner')
    df_final.rename(columns={'dat': 'DAT_Score'}, inplace=True)
    df_final = pd.merge(df_final, df_aut[['Pid', 'Total']], on='Pid', how='inner')
    df_final.rename(columns={'Total': 'AUT_Total'}, inplace=True)
    
    print("\n===============================================")
    print("GLOBAL STATISTICAL RESULTS")
    print("===============================================")
    
    # --- FAA Stats ---
    t_faa, p_faa = stats.ttest_rel(df_final['FAA_Treatment'], df_final['FAA_Control'])
    print(f"\n[PHASE 1] Frontal Alpha Asymmetry (Approach vs Withdrawal):")
    print(f"      -> t-stat: {t_faa:.3f}, p-value: {p_faa:.4f}")
    if p_faa < 0.05: print("      <<< SIGNIFICANT STRUCTURAL WITHDRAWAL DETECTED! <<<")
    generate_paired_offloading_plot(df_final, 'FAA', t_faa, p_faa, "Frontal Alpha Asymmetry [ln(F4) - ln(F3)]", "Global_FAA_Drive", output_dir)
        
    # --- Universal Dynamic Coherence Stats ---
    print(f"\n[PHASE 2] Universal Brain Decoupling Scan (p < 0.05 Threshold):")
    
    sig_pairs_dict = {band: {} for band in BANDS.keys()}
    total_decouplings = {band: 0 for band in BANDS.keys()}
    
    for ch1, ch2 in all_pairs:
        for band in BANDS.keys():
            t_s, p_s = stats.ttest_rel(df_final[f'Coh_{band}_{ch1}_{ch2}_Treatment'], df_final[f'Coh_{band}_{ch1}_{ch2}_Control'])
            
            if p_s < 0.05:
                total_decouplings[band] += 1
                sig_pairs_dict[band][(ch1, ch2)] = t_s
                print(f"  [{band.upper()}] {ch1} -> {ch2}: t={t_s:.3f}, p={p_s:.4f} <<< SIGNIFICANT")
                
                # Plot only elite-tier significances
                if p_s < 0.01:
                    generate_paired_offloading_plot(df_final, f'Coh_{band}_{ch1}_{ch2}', t_s, p_s, 
                        f"{band} Coherence Phase-Lock: {ch1} to {ch2}", f"Global_{band}_{ch1}_{ch2}", output_dir)
                
    # Generate the top-down academic Network Topoplot figures dynamically for any activated band
    for band in BANDS.keys():
        if len(sig_pairs_dict[band]) > 0:
            generate_network_topoplot(samp_raw, sig_pairs_dict[band], f'{band} Band', output_dir)
                
    print(f"\n  -> Found Localized Network Shifts:")
    for b_name in BANDS.keys():
        print(f"     * {b_name}: {total_decouplings[b_name]} structurally shifting bridges.")

    # --- Predictors ---
    print(f"\n[PHASE 3] Psychometric Trait Prediction Vectors:")
    for metric in ['DAT_Score', 'AUT_Total']:
        r, p_pearson = stats.pearsonr(df_final[metric], df_final['FAA_Shift'])
        print(f"  {metric} vs Asymmetry Shift -> Pearson's r: {r:.3f}, p = {p_pearson:.4f}")

    print(f"\n[SUCCESS] Global statistical visualizations rendered to {output_dir}/")

if __name__ == "__main__":
    main()
