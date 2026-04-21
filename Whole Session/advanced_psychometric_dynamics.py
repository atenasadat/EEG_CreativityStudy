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

warnings.filterwarnings("ignore")

def generate_network_topoplot(raw, sig_pairs_dict, band_name, output_dir, prefix=""):
    """Draws red/blue lines onto an overhead 2D topographical brain array."""
    if not sig_pairs_dict: return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    safe_raw = raw.copy().drop_channels(['A1', 'A2', 'boundary'], on_missing='ignore')
    
    try:
        picks = mne.pick_types(safe_raw.info, eeg=True)
        pos = mne.channels.layout._find_topomap_coords(safe_raw.info, picks=picks)
        ch_names = [safe_raw.ch_names[p] for p in picks]
        ch_pos = {ch: pos[i] for i, ch in enumerate(ch_names)}
        
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brain_template.png')
        if os.path.exists(template_path):
            img = plt.imread(template_path)
            ax.imshow(img, extent=[-0.125, 0.125, -0.13, 0.13], zorder=0, alpha=0.85)
        else:
            mne.viz.plot_sensors(safe_raw.info, ch_type='eeg', axes=ax, show=False, show_names=False, title="")
            
        for ch, (x, y) in ch_pos.items():
            ax.scatter(x, y, color='black', s=45, zorder=2, edgecolors='white', linewidth=1)
            ax.text(x, y + 0.006, ch, fontsize=8, ha='center', va='bottom', zorder=6, fontweight='bold')
            
    except Exception as e:
        print(f"   [!] Failed to extract Topoplot: {e}")
        plt.close()
        return

    ax.set_title(f"{prefix} Participant Network: {band_name}\nRed = AI Hyper-Coupling | Blue = AI Decoupling", fontweight='bold', pad=20)
    
    for (ch1, ch2), t_stat in sig_pairs_dict.items():
        if ch1 not in ch_pos or ch2 not in ch_pos: continue
        x1, y1 = ch_pos[ch1]
        x2, y2 = ch_pos[ch2]
        
        color = 'firebrick' if t_stat > 0 else 'dodgerblue'
        lw = np.clip(abs(t_stat), 1.5, 6)
        
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=0.8, zorder=5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_Topographical_Network_{band_name}.png"), dpi=600)
    plt.close()

def main():
    base_path = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    project_root = "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/"
    output_dir = "advanced_psychometric_stats"
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    print("\n=======================================================")
    print("ADVANCED PSYCHOMETRIC NETWORK PARSER")
    print("=======================================================")

    df_dat = pd.read_csv(os.path.join(project_root, "DATScores.csv"))
    df_aut = pd.read_csv(os.path.join(project_root, "AUTScores.csv"))
    df_dat['Pid'] = df_dat['Pid'].astype(str).str.replace(' ', '')
    df_aut['Pid'] = df_aut['Pid'].astype(str).str.replace(' ', '')

    c_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'control' in f.lower()]
    t_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'treatment' in f.lower()]
    
    c_dict = {f.split('_')[0]: f for f in c_files}
    t_dict = {f.split('_')[0]: f for f in t_files}
    matched_pids = sorted(list(set(c_dict.keys()) & set(t_dict.keys())), key=lambda x: int(x.replace('P', '')))
    
    samp_raw = mne.io.read_raw_eeglab(os.path.join(base_path, c_dict[matched_pids[0]]), preload=False, verbose=False)
    eeg_chans = [ch for ch in samp_raw.ch_names if ch not in ['A1', 'A2', 'boundary']]
    all_pairs = list(itertools.combinations(eeg_chans, 2))
    
    # We restrict to Theta (Working Memory) and Gamma (Logic Structuring) for psychological tracking
    BANDS = {'Theta': (4.0, 8.0), 'Gamma': (35.0, 50.0)}

    results = []
    print(f"[SYSTEM] Extracting Baseline Coherence Arrays for N={len(matched_pids)}...")
    for pid in matched_pids:
        raw_c = mne.io.read_raw_eeglab(os.path.join(base_path, c_dict[pid]), preload=True, verbose=False)
        raw_t = mne.io.read_raw_eeglab(os.path.join(base_path, t_dict[pid]), preload=True, verbose=False)
        fs = raw_c.info['sfreq']
        
        # Pull raw voltages dynamically preventing MNE memory bottlenecks
        data_c = raw_c.get_data(picks=eeg_chans)
        data_t = raw_t.get_data(picks=eeg_chans)
        
        row = {'Pid': pid}
        for ch1, ch2 in all_pairs:
            idx1, idx2 = eeg_chans.index(ch1), eeg_chans.index(ch2)
            f_c, cxy_c = scipy.signal.coherence(data_c[idx1], data_c[idx2], fs=fs, nperseg=1024)
            f_t, cxy_t = scipy.signal.coherence(data_t[idx1], data_t[idx2], fs=fs, nperseg=1024)
            
            for band_name, (fmin, fmax) in BANDS.items():
                mask = (f_c >= fmin) & (f_c <= fmax)
                row[f'Coh_{band_name}_{ch1}_{ch2}_Control'] = np.mean(cxy_c[mask])
                row[f'Coh_{band_name}_{ch1}_{ch2}_Treatment'] = np.mean(cxy_t[mask])
        results.append(row)
        
    df_eeg = pd.DataFrame(results)
    df_final = pd.merge(df_eeg, df_dat[['Pid', 'dat']], on='Pid')
    df_final.rename(columns={'dat': 'DAT_Score'}, inplace=True)
    df_final = pd.merge(df_final, df_aut[['Pid', 'Fluency', 'Flexibility', 'Elaboration', 'Originality', 'Total']], on='Pid')
    df_final.rename(columns={'Total': 'AUT_Total'}, inplace=True)

    print("\n===============================================")
    print("[PHASE 1] MEDIAN-SPLIT DAT TOPOGRAPHICAL MAPPING")
    print("===============================================")
    dat_median = df_final['DAT_Score'].median()
    df_high_dat = df_final[df_final['DAT_Score'] >= dat_median]
    df_low_dat = df_final[df_final['DAT_Score'] < dat_median]
    
    print(f"  -> High DAT Group (N={len(df_high_dat)}): DAT >= {dat_median}")
    print(f"  -> Low DAT Group  (N={len(df_low_dat)}): DAT < {dat_median}\n")
    
    for df_sub, prefix in [(df_high_dat, "HighDAT"), (df_low_dat, "LowDAT")]:
        sig_pairs_dict = {band: {} for band in BANDS.keys()}
        for ch1, ch2 in all_pairs:
            for band in BANDS.keys():
                t_s, p_s = stats.ttest_rel(df_sub[f'Coh_{band}_{ch1}_{ch2}_Treatment'], df_sub[f'Coh_{band}_{ch1}_{ch2}_Control'])
                if p_s < 0.05:
                    sig_pairs_dict[band][(ch1, ch2)] = t_s
                    
        for band in BANDS.keys():
            print(f"  [{prefix}] {band} Network: {len(sig_pairs_dict[band])} significant shifting connections.")
            if len(sig_pairs_dict[band]) > 0:
                generate_network_topoplot(samp_raw, sig_pairs_dict[band], f'{band} Band', output_dir, prefix=prefix)

    print("\n===============================================")
    print("[PHASE 2] EXECUTIVE DECOUPLING MAGNITUDE CORRELATION")
    print("===============================================")
    # Magnitude defined as Control - AI (Positive magnitude means coupling crashed significantly when using AI)
    df_final['Mag_Fz_Cz'] = df_final['Coh_Gamma_Fz_Cz_Control'] - df_final['Coh_Gamma_Fz_Cz_Treatment']
    df_final['Mag_F3_Fz'] = df_final['Coh_Gamma_F3_Fz_Control'] - df_final['Coh_Gamma_F3_Fz_Treatment']

    for metric_name, col_name in [("Fz-Cz (Motor Syntax)", "Mag_Fz_Cz"), ("F3-Fz (Logic Formulation)", "Mag_F3_Fz")]:
        r, p_val = stats.pearsonr(df_final['DAT_Score'], df_final[col_name])
        print(f"  DAT vs {metric_name} Decoupling Drop Magnitude:")
        print(f"     -> r = {r:.3f}, p = {p_val:.4f} " + ("<<< SIGNIFICANT" if p_val < 0.05 else ""))
        
        plt.figure(figsize=(7, 5))
        sns.regplot(data=df_final, x='DAT_Score', y=col_name, scatter_kws={'s':60, 'alpha':0.7}, line_kws={'color':'red'})
        plt.title(f"DAT vs Gamma {metric_name} Decoupling\nPearson r={r:.3f}, p={p_val:.4f}", fontweight='bold')
        plt.xlabel("DAT Score (Innate Divergent Creativity Traits)")
        plt.ylabel("Decoupling Magnitude (Control - AI Assisted)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Scatter_DAT_vs_{col_name}.png"), dpi=600)
        plt.close()

    print("\n===============================================")
    print("[PHASE 3] AUT SUB-COMPONENT HEATMAP DISCOVERY")
    print("===============================================")
    aut_cols = ['Fluency', 'Flexibility', 'Elaboration', 'Originality']
    
    for band in BANDS.keys():
        print(f"\n  Top Predictors for {band} Shift Mapping:")
        for aut_sub in aut_cols:
            best_r, best_p, best_pair = 0, 1.0, None
            for ch1, ch2 in all_pairs:
                # Calculate the raw absolute topological shift bridge
                shift_mag = df_final[f'Coh_{band}_{ch1}_{ch2}_Treatment'] - df_final[f'Coh_{band}_{ch1}_{ch2}_Control']
                r, p_val = stats.pearsonr(df_final[aut_sub], shift_mag)
                if p_val < best_p:
                    best_r, best_p, best_pair = r, p_val, f"{ch1}-{ch2}"
            if best_p < 0.05:
                print(f"     * {aut_sub} heavily predicts '{best_pair}' network mapping (r={best_r:.3f}, p={best_p:.4f})")
            else:
                print(f"     * {aut_sub} -> No structural network correlations found under LLM logic parameters.")

    print(f"\n[SUCCESS] Advanced Psychometric processing completed. Results dumped to {output_dir}/")

if __name__ == "__main__":
    main()
