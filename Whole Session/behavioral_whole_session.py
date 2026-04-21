import os
import glob
import numpy as np
import pandas as pd
import mne
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress messy MNE warnings for clean terminal output
warnings.filterwarnings("ignore")

# Define target regions based on theoretical mappings
theta_frontal = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8']
alpha_rpo = ['P4', 'P8', 'O2'] # Right Parieto-Occipital
gamma_rst = ['T8', 'P8']       # Right Superior Temporal

def get_mean_psd(raw, channels, fmin, fmax):
    """Calculates continuous Welch PSD power for a specific channel set and frequency band."""
    raw_roi = raw.copy().pick(channels)
    
    # Using modern MNE API to calculate Power Spectral Density over the entire session length
    # Note: Using default Welch parameters (2048 FFT length) which is standard for continuous EEG
    spectrum = raw_roi.compute_psd(method='welch', fmin=fmin, fmax=fmax, picks='eeg', verbose=False)
    
    # Return the mean absolute power across all channels, frequencies, and epochs in the requested range
    return np.mean(spectrum.get_data())

def generate_paired_offloading_plot(df, metric_base, t_val, p_val, title_override, out_file, output_dir):
    df_melt = pd.melt(df, id_vars=['Pid'], 
                      value_vars=[f'{metric_base}_Control', f'{metric_base}_Treatment'],
                      var_name='Condition', value_name='Signal Power')
    df_melt['Condition'] = df_melt['Condition'].map({f'{metric_base}_Control': 'Unassisted', f'{metric_base}_Treatment': 'LLM Assisted'})
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_melt, x='Condition', y='Signal Power', color='grey', boxprops=dict(alpha=0.3))
    sns.swarmplot(data=df_melt, x='Condition', y='Signal Power', color='black', alpha=0.5)
    
    for pid in df['Pid'].unique():
        p_c = df_melt[(df_melt['Pid'] == pid) & (df_melt['Condition'] == 'Unassisted')]['Signal Power'].values[0]
        p_t = df_melt[(df_melt['Pid'] == pid) & (df_melt['Condition'] == 'LLM Assisted')]['Signal Power'].values[0]
        plt.plot(['Unassisted', 'LLM Assisted'], [p_c, p_t], color='gray', alpha=0.4, linestyle='--')
        
    sig_mark = "***" if p_val < 0.05 else ("*" if p_val < 0.1 else "n.s.")
    plt.title(f"{title_override}\nt-stat: {t_val:.3f} | p-value: {p_val:.4f} ({sig_mark})", fontweight='bold')
    plt.ylabel("PSD Power (\u03bcV\u00b2/Hz)") # Standard Welch unit
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{out_file}.png"), dpi=600)
    plt.close()

def main():
    # Workspace Configuration
    base_path = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    project_root = "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/"
    output_dir = "whole_session_stats"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("\n=======================================================")
    print("CONTINUOUS WHOLE-SESSION EEG SPECTRAL ANALYSIS PIPELINE")
    print("=======================================================")

    # Load Behavioral Scores
    try:
        df_dat = pd.read_csv(os.path.join(project_root, "DATScores.csv"))
        df_aut = pd.read_csv(os.path.join(project_root, "AUTScores.csv"))
        df_dat['Pid'] = df_dat['Pid'].astype(str).str.replace(' ', '')
        df_aut['Pid'] = df_aut['Pid'].astype(str).str.replace(' ', '')
    except FileNotFoundError:
        print("CRITICAL ERROR: Behavioral CSVs not found in project root.")
        return

    # Match Particpants (Trigger-independent)
    c_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'control' in f.lower()]
    t_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'treatment' in f.lower()]
    
    c_dict = {f.split('_')[0]: f for f in c_files}
    t_dict = {f.split('_')[0]: f for f in t_files}
    matched_pids = sorted(list(set(c_dict.keys()) & set(t_dict.keys())), key=lambda x: int(x.replace('P', '')))
    
    print(f"Matched Participant Array Size: N={len(matched_pids)}\n")
    
    results = []
    
    for pid in matched_pids:
        print(f"  -> Processing Whole Session: {pid}...")
        
        # Load continuous raw data
        raw_c = mne.io.read_raw_eeglab(os.path.join(base_path, c_dict[pid]), preload=True, verbose=False)
        raw_t = mne.io.read_raw_eeglab(os.path.join(base_path, t_dict[pid]), preload=True, verbose=False)
        
        row = {'Pid': pid}
        
        # --- FEATURE 1: Generalized Working Memory Burden (Frontal Theta) ---
        row['Theta_Frontal_Control'] = get_mean_psd(raw_c, theta_frontal, 4, 8)
        row['Theta_Frontal_Treatment'] = get_mean_psd(raw_t, theta_frontal, 4, 8)
        row['Theta_Frontal_Diff'] = row['Theta_Frontal_Treatment'] - row['Theta_Frontal_Control']
        
        # --- FEATURE 2: Generalized Visual/Ideation Activity (R.P.O Alpha) ---
        row['Alpha_RPO_Control'] = get_mean_psd(raw_c, alpha_rpo, 8, 13)
        row['Alpha_RPO_Treatment'] = get_mean_psd(raw_t, alpha_rpo, 8, 13)
        row['Alpha_RPO_Diff'] = row['Alpha_RPO_Treatment'] - row['Alpha_RPO_Control']
        
        # --- FEATURE 3: Generalized Structural Spark (R.S.T Gamma) ---
        row['Gamma_RST_Control'] = get_mean_psd(raw_c, gamma_rst, 35, 50)
        row['Gamma_RST_Treatment'] = get_mean_psd(raw_t, gamma_rst, 35, 50)
        row['Gamma_RST_Diff'] = row['Gamma_RST_Treatment'] - row['Gamma_RST_Control']
        
        # --- Multi-Band Discrete Electrodes ---
        for ch in theta_frontal:
            row[f'Theta_{ch}_C'] = get_mean_psd(raw_c, [ch], 4, 8)
            row[f'Theta_{ch}_T'] = get_mean_psd(raw_t, [ch], 4, 8)
            
        for ch in alpha_rpo:
            row[f'Alpha_{ch}_C'] = get_mean_psd(raw_c, [ch], 8, 13)
            row[f'Alpha_{ch}_T'] = get_mean_psd(raw_t, [ch], 8, 13)
            
        for ch in gamma_rst:
            row[f'Gamma_{ch}_C'] = get_mean_psd(raw_c, [ch], 35, 50)
            row[f'Gamma_{ch}_T'] = get_mean_psd(raw_t, [ch], 35, 50)
            
        results.append(row)
        
    df_eeg = pd.DataFrame(results)
    
    # Merge Psychometric Scores
    df_final = pd.merge(df_eeg, df_dat[['Pid', 'dat']], on='Pid', how='inner')
    df_final.rename(columns={'dat': 'DAT_Score'}, inplace=True)
    
    df_final = pd.merge(df_final, df_aut[['Pid', 'Total']], on='Pid', how='inner')
    df_final.rename(columns={'Total': 'AUT_Total'}, inplace=True)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    print("\n===============================================")
    print("STATISTICAL RESULTS CACHE (WHOLE SESSION)")
    print("===============================================")
    
    # -------------------------------------------------------------------------
    # ANALYSIS 1: Psychometric Interaction (Frontal Theta Working Memory)
    # -------------------------------------------------------------------------
    for metric in ['DAT_Score', 'AUT_Total']:
        r, p_pearson = stats.pearsonr(df_final[metric], df_final['Theta_Frontal_Diff'])
        rho, p_spearman = stats.spearmanr(df_final[metric], df_final['Theta_Frontal_Diff'])
        
        print(f"[Interaction] {metric} vs AI Frontal Theta Shift (Generalized Burden):")
        print(f"      -> Pearson's r: {r:.3f}, p = {p_pearson:.4f}")
        print(f"      -> Spearman's rho: {rho:.3f}, p = {p_spearman:.4f}")
        
        plt.figure(figsize=(7, 5))
        sns.regplot(data=df_final, x=metric, y='Theta_Frontal_Diff', color='#1f77b4', scatter_kws={'s': 80})
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Brain-Behavior Interaction: Frontal Theta Shift vs {metric}\nPearson r={r:.3f}, p={p_pearson:.4f}", fontweight='bold')
        plt.ylabel("Frontal Theta PSD Shift (LLM - Alone)")
        plt.xlabel(f"Psychometric Score ({metric})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Correlation_Theta_{metric}.png"), dpi=600)
        plt.close()

    # -------------------------------------------------------------------------
    # ANALYSIS 2: Paired Theory Blocks
    # -------------------------------------------------------------------------
    def test_and_plot(name_prefix, col_base, description):
        t_stat, p_val = stats.ttest_rel(df_final[f'{col_base}_Treatment'], df_final[f'{col_base}_Control'])
        
        sig_str = " *** SIGNIFICANT ***" if p_val < 0.05 else (" * TRENDING *" if p_val < 0.1 else "")
        print(f"\n[{name_prefix}] {description}")
        print(f"   -> t-stat = {t_stat:.3f}, p-value = {p_val:.4f}{sig_str}")
        
        generate_paired_offloading_plot(df_final, col_base, t_stat, p_val, description, f"WholeSession_{col_base}", output_dir)
        
    test_and_plot("Working Memory", "Theta_Frontal", "Generalized Frontal Theta PSD (4-8 Hz)")
    test_and_plot("Visual Ideation", "Alpha_RPO", "Generalized Right Parieto-Occipital Alpha PSD (8-13 Hz)")
    test_and_plot("Insight Spark", "Gamma_RST", "Generalized Right Superior Temporal Gamma PSD (35-50 Hz)")

    # -------------------------------------------------------------------------
    # ANALYSIS 3: Discrete High-Resolution Electrode Mapping
    # -------------------------------------------------------------------------
    print(f"\n=======================================================")
    print(f"[Discrete Mapping] Single Channel Sub-Network Power Shifts")
    print(f"=======================================================")
    
    def scan_discrete_nodes(ch_list, band_pfx, name_pfx):
        found_any = False
        for ch in ch_list:
            col_c = f'{band_pfx}_{ch}_C'
            col_t = f'{band_pfx}_{ch}_T'
            t_ch, p_ch = stats.ttest_rel(df_final[col_t], df_final[col_c])
            
            marker = "<<< SIGNIFICANT DIFFERENCE (p<0.05)!" if p_ch < 0.05 else ("< Trending (p<0.1)" if p_ch < 0.1 else "")
            if marker: found_any = True
                
            print(f"  [{ch} {name_pfx}] -> t={t_ch:6.3f}, p={p_ch:.4f} {marker}")
            if p_ch < 0.1:
                df_temp = df_final.copy()
                df_temp.rename(columns={col_c: f'{band_pfx}_{ch}_Control', col_t: f'{band_pfx}_{ch}_Treatment'}, inplace=True)
                generate_paired_offloading_plot(
                    df_temp, f"{band_pfx}_{ch}", t_ch, p_ch, 
                    f"Single Node Whole Session {name_pfx}: {ch}", 
                    f"WholeSession_{band_pfx}_{ch}", output_dir)
        if not found_any: print(f"  >>> No native significance found in {name_pfx}.")

    print("\n--- 1. Frontal Theta Power (Working Memory) ---")
    scan_discrete_nodes(theta_frontal, 'Theta', 'Frontal Theta')
    
    print("\n--- 2. Right Parieto-Occipital Alpha (Visual / Brainstorming) ---")
    scan_discrete_nodes(alpha_rpo, 'Alpha', 'R.P.O. Alpha')
    
    print("\n--- 3. Right Superior Temporal Gamma (The Aha! Spark) ---")
    scan_discrete_nodes(gamma_rst, 'Gamma', 'R.S.T. Gamma')
    
    print(f"\nHighly Detailed Statistical Visualizations exported natively to {output_dir}/")

if __name__ == "__main__":
    main()
