import os
import glob
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def get_files_by_prefix(base_path, prefix_tag="postcleaning"):
    """Isolate matched Control and Treatment `.set` files accurately mapping to Participant IDs."""
    control_files = []
    treatment_files = []
    for f in os.listdir(base_path):
        if f.endswith('.set') and prefix_tag in f:
            if 'control' in f.lower():
                control_files.append(f)
            elif 'treatment' in f.lower():
                treatment_files.append(f)
    return control_files, treatment_files

def match_participants(control_files, treatment_files):
    """Ensure we only process participants who have both conditions for paired statistical power."""
    c_dict = {f.split('_')[0]: f for f in control_files}
    t_dict = {f.split('_')[0]: f for f in treatment_files}
    
    matched_pids = sorted(list(set(c_dict.keys()) & set(t_dict.keys())))
    matched_pids.sort(key=lambda x: int(x.replace('P', '')))
    
    return matched_pids, c_dict, t_dict

def get_mean_amplitude(epochs, channels, tmin, tmax):
    """Isolate scalar Evoked Voltage over a specific targeted Topographical sub-cluster window."""
    evoked = epochs.average().pick(channels)
    evoked.crop(tmin, tmax)
    return np.mean(evoked.data)

def get_mean_alpha_tfr(epochs, channels, tmin, tmax):
    """Rapid Morlet extraction explicitly tuned to the Alpha-band Frontal subnetwork."""
    # Create a native copy to prevent destructive in-place dropping of channels for subsequent phases
    epochs_roi = epochs.copy().pick(channels)
    
    # Strictly target internal creative cognitive bandwidth (8-13 Hz M/Alpha)
    freqs = np.arange(8, 14, 1.0)
    
    # We do NOT apply a log-ratio baseline here because we are explicitly comparing 
    # absolute background internal states at baseline, rather than transient edge changes.
    tfr = epochs_roi.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs / 2.0, return_itc=False, average=True, verbose=False)
    
    tfr.crop(tmin, tmax)
    # Mean across all sensors (channels), frequencies, and time segments in this window
    return np.mean(tfr.data)

def get_mean_gamma_tfr(epochs, channels, tmin, tmax):
    """Isolates the high-frequency Gamma burst signifying the absolute 'Spark' of structural ideation."""
    epochs_roi = epochs.copy().pick(channels)
    
    # High-frequency band (35-50 Hz) target right before neural completion
    freqs = np.arange(35, 51, 2.0)
    
    tfr = epochs_roi.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs / 4.0, return_itc=False, average=True, verbose=False)
    tfr.crop(tmin, tmax)
    return np.mean(tfr.data)

def get_mean_theta_tfr(epochs, channels, tmin, tmax):
    """Extracts slow-wave Theta power (4-8 Hz), often linked to demanding working memory and internal focus."""
    epochs_roi = epochs.copy().pick(channels)
    
    freqs = np.arange(4, 9, 1.0)
    tfr = epochs_roi.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs / 2.0, return_itc=False, average=True, verbose=False)
    tfr.crop(tmin, tmax)
    return np.mean(tfr.data)

def main():
    # Workspace Configuration
    base_path = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    output_dir = "behavioral_stats"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ingest Psychometric Behavior Metrics
    # Moving up one folder level assuming behavioral CSVs are saved in the project root
    project_root = "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/"
    dat_df = pd.read_csv(os.path.join(project_root, 'DATScores.csv'))
    aut_df = pd.read_csv(os.path.join(project_root, 'AUTScores.csv'))
    
    # Aggregating metrics across psychometric tables
    df_beh = pd.merge(dat_df[['Pid', 'dat']], aut_df[['Pid', 'Total', 'Fluency', 'Flexibility', 'Elaboration', 'Originality']], on='Pid', how='inner')
    df_beh.rename(columns={'dat': 'DAT_Score', 'Total': 'AUT_Total'}, inplace=True)
    
    # 2. Ingest Neurological Data structures
    control_files, treatment_files = get_files_by_prefix(base_path)
    matched_pids, c_dict, t_dict = match_participants(control_files, treatment_files)
    
    # 19 MNE System 10-20 Standard Channels
    standard_channels = [
        'P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1', 
        'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'P8', 'T8', 'Pz'
    ]
    
    trigger_id = '100'
    
    # Define Core ROIs to match Permutation Cluster 51 & Cognitive Offloading theories
    cluster51_chans = ['P3', 'C3', 'F3']
    frontal_chans = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8']
    gamma_chans = ['T8', 'P8'] # Beeman's Right Temporal Insight ROI
    prefrontal_chans = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4'] # User's Prefrontal Gamma ROI
    # User's targeted discrete electrodes for individual-level Gamma significance mapping
    discrete_gamma_chans = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'O1', 'O2', 'P3', 'Pz', 'P4', 'P7', 'P8']
    
    # New Multi-Band Theoretical Regions:
    theta_frontal = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8']
    alpha_rpo = ['P4', 'P8', 'O2'] # Right Parieto-Occipital
    gamma_rst = ['T8', 'P8'] # Right Superior Temporal (Gamma)
    
    results = []
    
    print("\nPhase 1 & 2: Mathematically extracting Neural Voltage Transients and Alpha Frequency Oscillations...")
    print(f"Matched Participant Array Size: N={len(matched_pids)}")
    
    for pid in matched_pids:
        print(f"  -> Extrapolating Subject: {pid}...")
        
        # Parse MNE Raw `.set` Control Matrix
        raw_c = mne.io.read_raw_eeglab(os.path.join(base_path, c_dict[pid]), preload=True, verbose=False)
        raw_c.pick(standard_channels)
        events_c, event_id_c = mne.events_from_annotations(raw_c, verbose=False)
        if trigger_id not in event_id_c: 
            print(f"Warning: Subject {pid} missing trigger in Control.")
            continue
        epochs_c = mne.Epochs(raw_c, events_c, event_id_c[trigger_id], tmin=-2.0, tmax=1.0, baseline=(-0.2, 0), preload=True, verbose=False)
        
        # Parse MNE Raw `.set` Treatment Matrix
        raw_t = mne.io.read_raw_eeglab(os.path.join(base_path, t_dict[pid]), preload=True, verbose=False)
        raw_t.pick(standard_channels)
        events_t, event_id_t = mne.events_from_annotations(raw_t, verbose=False)
        if trigger_id not in event_id_t: 
            print(f"Warning: Subject {pid} missing trigger in Treatment.")
            continue
        epochs_t = mne.Epochs(raw_t, events_t, event_id_t[trigger_id], tmin=-2.0, tmax=1.0, baseline=(-0.2, 0), preload=True, verbose=False)
        
        # --- FEATURE 1: Cluster 51 Target Voltage ---
        # Time Window exactly mirroring the significance bound (-450ms to -297ms)
        v_c = get_mean_amplitude(epochs_c, cluster51_chans, -0.450, -0.297)
        v_t = get_mean_amplitude(epochs_t, cluster51_chans, -0.450, -0.297)
        # Compute "Leveling/Synergy Shift" (How much the AI changed the brain state)
        v_diff = v_t - v_c 
        
        # --- FEATURE 2: Frontal Alpha Offloading ---
        # Immediate pre-insight window capturing Internal Brainstorming state
        a_c = get_mean_alpha_tfr(epochs_c, frontal_chans, -0.5, 0.0)
        a_t = get_mean_alpha_tfr(epochs_t, frontal_chans, -0.5, 0.0)
        
        # --- FEATURE 3: Right Temporal Gamma Burst ---
        # Splitting the absolute burst window (-0.3s to 0.0s) immediately prior to Aha!
        g_c = get_mean_gamma_tfr(epochs_c, gamma_chans, -0.3, 0.0)
        g_t = get_mean_gamma_tfr(epochs_t, gamma_chans, -0.3, 0.0)
        
        # --- FEATURE 4: Prefrontal Gamma Burst ---
        pg_c = get_mean_gamma_tfr(epochs_c, prefrontal_chans, -0.3, 0.0)
        pg_t = get_mean_gamma_tfr(epochs_t, prefrontal_chans, -0.3, 0.0)
        
        row = {
            'Pid': pid,
            'Cluster51_Control': v_c,
            'Cluster51_Treatment': v_t,
            'Cluster51_Diff': v_diff,
            'Alpha_Control': a_c,
            'Alpha_Treatment': a_t,
            'Alpha_Diff': a_t - a_c,
            'Gamma_Control': g_c,
            'Gamma_Treatment': g_t,
            'Gamma_Diff': g_t - g_c,
            'PrefrontalGamma_Control': pg_c,
            'PrefrontalGamma_Treatment': pg_t
        }
        
        # --- FEATURE 5: Discrete Channel Gamma Burst Scan ---
        for ch in discrete_gamma_chans:
            # Pre-Insight Gamma
            val_c = get_mean_gamma_tfr(epochs_c, [ch], -0.3, 0.0)
            val_t = get_mean_gamma_tfr(epochs_t, [ch], -0.3, 0.0)
            row[f'Gamma_{ch}_Control'] = val_c
            row[f'Gamma_{ch}_Treatment'] = val_t
            
            # Post-Insight Gamma (0.0s to 0.45s to avoid edge artifacts of 0.5s epoch boundary)
            post_c = get_mean_gamma_tfr(epochs_c, [ch], 0.0, 0.45)
            post_t = get_mean_gamma_tfr(epochs_t, [ch], 0.0, 0.45)
            row[f'PostGamma_{ch}_Control'] = post_c
            row[f'PostGamma_{ch}_Treatment'] = post_t
            
        # --- FEATURE 6: Theoretically Grounded Multi-Band Single Electrodes ---
        # 1. Frontal Theta (Working Memory)
        for ch in theta_frontal:
            row[f'Theta_{ch}_Pre_C'] = get_mean_theta_tfr(epochs_c, [ch], -0.5, 0.0)
            row[f'Theta_{ch}_Pre_T'] = get_mean_theta_tfr(epochs_t, [ch], -0.5, 0.0)
            row[f'Theta_{ch}_Post_C'] = get_mean_theta_tfr(epochs_c, [ch], 0.0, 0.45)
            row[f'Theta_{ch}_Post_T'] = get_mean_theta_tfr(epochs_t, [ch], 0.0, 0.45)
            
        # 2. Right Parieto-Occipital Alpha (Visual / Default Mode suppression)
        for ch in alpha_rpo:
            row[f'Alpha_{ch}_Pre_C'] = get_mean_alpha_tfr(epochs_c, [ch], -0.5, 0.0)
            row[f'Alpha_{ch}_Pre_T'] = get_mean_alpha_tfr(epochs_t, [ch], -0.5, 0.0)
            row[f'Alpha_{ch}_Post_C'] = get_mean_alpha_tfr(epochs_c, [ch], 0.0, 0.45)
            row[f'Alpha_{ch}_Post_T'] = get_mean_alpha_tfr(epochs_t, [ch], 0.0, 0.45)
            
        # 3. Right Superior Temporal Gamma (Spark)
        for ch in gamma_rst:
            row[f'RSTGamma_{ch}_Pre_C'] = get_mean_gamma_tfr(epochs_c, [ch], -0.3, 0.0)
            row[f'RSTGamma_{ch}_Pre_T'] = get_mean_gamma_tfr(epochs_t, [ch], -0.3, 0.0)
            row[f'RSTGamma_{ch}_Post_C'] = get_mean_gamma_tfr(epochs_c, [ch], 0.0, 0.45)
            row[f'RSTGamma_{ch}_Post_T'] = get_mean_gamma_tfr(epochs_t, [ch], 0.0, 0.45)
            
        results.append(row)
        
    df_eeg = pd.DataFrame(results)
    
    # Merge human-behavior scale metrics directly into exact brain EEG scalar aggregates
    df_final = pd.merge(df_beh, df_eeg, on='Pid', how='inner')
    
    print("\n===============================================")
    print("STATISTICAL RESULTS CACHE & NULL-HYPOTHESES")
    print("===============================================")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # -------------------------------------------------------------------------
    # ANALYSIS 1: Pearson Correlation (Interaction between AI effect and Psychometrics)
    # -------------------------------------------------------------------------
    for metric in ['DAT_Score', 'AUT_Total']:
        r, p_pearson = stats.pearsonr(df_final[metric], df_final['Cluster51_Diff'])
        rho, p_spearman = stats.spearmanr(df_final[metric], df_final['Cluster51_Diff'])
        
        print(f"[Interaction] {metric} vs AI Brain-Leveling Shift (Cluster 51):")
        print(f"      -> Pearson's r: {r:.3f}, p = {p_pearson:.4f}")
        print(f"      -> Spearman's rho: {rho:.3f}, p = {p_spearman:.4f}")
        
        plt.figure(figsize=(7, 5))
        sns.regplot(data=df_final, x=metric, y='Cluster51_Diff', color='#8C564B', scatter_kws={'s': 80})
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Brain-Behavior Interaction: AI Voltage Shift vs {metric}\nPearson's r = {r:.3f}, p-val = {p_pearson:.4f}", fontweight='bold')
        
        plt.ylabel("Cluster 51 ERP Shift (LLM - Alone)")
        plt.xlabel(f"Psychometric Score ({metric})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Correlation_Cluster51_{metric}.png"), dpi=600)
        plt.close()
        
    # -------------------------------------------------------------------------
    # ANALYSIS 3: Cognitive Offloading Paired T-Test
    # -------------------------------------------------------------------------
    t_stat, p_val = stats.ttest_rel(df_final['Alpha_Treatment'], df_final['Alpha_Control'])
    
    print(f"\n[Cognitive Offloading Theory] Frontal Alpha Power (-0.5s to 0s):")
    print(f"   -> t-stat = {t_stat:.3f}, p-value = {p_val:.4f}")
    if p_val < 0.05: print("   >>> RESULT: Statistically Significant shift in internal generative process!")
    else: print("   >>> RESULT: No statistically significant Alpha offloading detected.")
        
    # --- Shared Boxplot Configurator ---
    def generate_paired_offloading_plot(df_focused, metric_name, t, p, title, out_name):
        plt.figure(figsize=(6, 6))
        
        # Safe dataframe handling
        df_melt = df_focused[['Pid', f'{metric_name}_Control', f'{metric_name}_Treatment']].melt(id_vars='Pid', var_name='Condition', value_name='Power')
        df_melt['Condition'] = df_melt['Condition'].map({f'{metric_name}_Control': 'Unassisted (Alone)', f'{metric_name}_Treatment': 'LLM Assisted'})
        
        # Violin & Dots overlay. Hue mapping resolves seaborn deprecation warnings!
        sns.violinplot(data=df_melt, x='Condition', y='Power', hue='Condition', palette={'Unassisted (Alone)': '#D25353', 'LLM Assisted': '#5A9CB5'}, legend=False)
        sns.stripplot(data=df_melt, x='Condition', y='Power', color='black', alpha=0.5, size=6, jitter=False)
        
        # Paired Lines tracking individual subjects across the condition shift
        pivoted = df_melt.pivot(index='Pid', columns='Condition', values='Power')
        for i in range(len(pivoted)):
            plt.plot([0, 1], [pivoted.iloc[i]['Unassisted (Alone)'], pivoted.iloc[i]['LLM Assisted']], color='gray', alpha=0.3, zorder=0)
            
        plt.title(f"{title}\nPaired t-test: t = {t:.3f}, p = {p:.4f}", fontweight='bold')
        plt.ylabel(f"{metric_name} Band Power")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{out_name}.png"), dpi=600)
        plt.close()

    # Generate Alpha Chart
    generate_paired_offloading_plot(df_final, "Alpha", t_stat, p_val, "Frontal Alpha Cognitive Offloading (-0.5s to 0s)", "Cognitive_Offloading_Alpha")

    # -------------------------------------------------------------------------
    # ANALYSIS 4: Right Temporal Gamma "Aha!" Spark
    # -------------------------------------------------------------------------
    t_gamma, p_gamma = stats.ttest_rel(df_final['Gamma_Treatment'], df_final['Gamma_Control'])
    
    print(f"\n[Aha! Spark] Right Temporal Gamma Burst (-0.3s to 0s):")
    print(f"   -> t-stat = {t_gamma:.3f}, p-value = {p_gamma:.4f}")
    
    # Generate Temporal Gamma Chart
    generate_paired_offloading_plot(df_final, "Gamma", t_gamma, p_gamma, "Right Temporal Gamma Burst (-0.3s to 0s)", "Cognitive_Offloading_Gamma")

    # -------------------------------------------------------------------------
    # ANALYSIS 5: Prefrontal Gamma Burst
    # -------------------------------------------------------------------------
    t_pfgamma, p_pfgamma = stats.ttest_rel(df_final['PrefrontalGamma_Treatment'], df_final['PrefrontalGamma_Control'])
    
    print(f"\n[Prefrontal Spark] Prefrontal Gamma Burst (-0.3s to 0s):")
    print(f"   -> t-stat = {t_pfgamma:.3f}, p-value = {p_pfgamma:.4f}")
    if p_pfgamma < 0.1: print("   >>> RESULT: Trending effect found in Prefrontal Gamma!")
    
    # Generate Prefrontal Gamma Chart
    generate_paired_offloading_plot(df_final, "PrefrontalGamma", t_pfgamma, p_pfgamma, "Prefrontal Gamma Burst (-0.3s to 0s)", "Cognitive_Offloading_PrefrontalGamma")
    
    # -------------------------------------------------------------------------
    # ANALYSIS 6: Discrete High-Resolution Electrode Mapping (Gamma)
    # -------------------------------------------------------------------------
    print(f"\n=======================================================")
    print(f"[Discrete Mapping] High-Res Single Channel Gamma Scans (-0.3s to 0s)")
    print(f"=======================================================")
    
    found_any = False
    for ch in discrete_gamma_chans:
        col_c = f'Gamma_{ch}_Control'
        col_t = f'Gamma_{ch}_Treatment'
        t_ch, p_ch = stats.ttest_rel(df_final[col_t], df_final[col_c])
        
        marker = ""
        if p_ch < 0.05: 
            marker = "<<< SIGNIFICANT DIFFERENCE (p<0.05)!"
            found_any = True
        elif p_ch < 0.1:
            marker = "< Trending (p<0.1)"
            
        print(f"  [{ch} Pre-Insight Gamma] -> t={t_ch:6.3f}, p={p_ch:.4f} {marker}")
        
        # We will auto-generate charts specifically for channels that drop below 0.1
        if p_ch < 0.1:
            generate_paired_offloading_plot(
                df_final, f"Gamma_{ch}", t_ch, p_ch, 
                f"Single Node Pre-Insight Gamma: {ch} (-0.3s to 0s)", 
                f"Cognitive_Offloading_PreGamma_{ch}")
            
    if not found_any:
        print("\n  >>> RESULT: No individual electrodes hit native p<0.05 structural significance in PRE-insight.")

    # -------------------------------------------------------------------------
    # ANALYSIS 7: POST-INSIGHT Discrete Mapping (0.0s to 0.45s)
    # -------------------------------------------------------------------------
    print(f"\n=======================================================")
    print(f"[Discrete Mapping] POST-INSIGHT Gamma Scans (0.0s to +0.45s)")
    print(f"=======================================================")
    
    found_any_post = False
    for ch in discrete_gamma_chans:
        col_c = f'PostGamma_{ch}_Control'
        col_t = f'PostGamma_{ch}_Treatment'
        t_ch, p_ch = stats.ttest_rel(df_final[col_t], df_final[col_c])
        
        marker = ""
        if p_ch < 0.05: 
            marker = "<<< SIGNIFICANT DIFFERENCE (p<0.05)!"
            found_any_post = True
        elif p_ch < 0.1:
            marker = "< Trending (p<0.1)"
            
        print(f"  [{ch} POST-Insight Gamma] -> t={t_ch:6.3f}, p={p_ch:.4f} {marker}")
        
        # We will auto-generate charts specifically for channels that drop below 0.1
        if p_ch < 0.1:
            generate_paired_offloading_plot(
                df_final, f"PostGamma_{ch}", t_ch, p_ch, 
                f"Single Node POST-Insight Gamma: {ch} (0.0s to 0.45s)", 
                f"Cognitive_Offloading_PostGamma_{ch}")
            
    if not found_any_post:
        print("\n  >>> RESULT: No individual electrodes hit native p<0.05 structural significance in POST-insight.")

    # -------------------------------------------------------------------------
    # ANALYSIS 8: Targeted Multi-Band Multi-Region Discrete Mapping
    # -------------------------------------------------------------------------
    def scan_and_print_discrete_nodes(ch_list, band_pfx, name_pfx, time_pfx):
        found_any = False
        for ch in ch_list:
            col_c = f'{band_pfx}_{ch}_{time_pfx}_C'
            col_t = f'{band_pfx}_{ch}_{time_pfx}_T'
            t_ch, p_ch = stats.ttest_rel(df_final[col_t], df_final[col_c])
            
            marker = ""
            if p_ch < 0.05: 
                marker = "<<< SIGNIFICANT DIFFERENCE (p<0.05)!"
                found_any = True
            elif p_ch < 0.1:
                marker = "< Trending (p<0.1)"
                
            print(f"  [{ch} {name_pfx}] -> t={t_ch:6.3f}, p={p_ch:.4f} {marker}")
            if p_ch < 0.1:
                generate_paired_offloading_plot(df_final, f"{band_pfx}_{ch}_{time_pfx}", t_ch, p_ch, 
                f"Single Node {name_pfx}: {ch}", f"Cognitive_Offloading_{name_pfx.replace(' ', '')}_{ch}")
        if not found_any: print(f"  >>> No native significance found in {name_pfx}.")
        
    print(f"\n=======================================================")
    print(f"[Advanced Mapping] multi-Band Theoretical Regions")
    print(f"=======================================================")
    
    print("\n--- 1. Frontal Theta Power (Working Memory) ---")
    print(" PRE-Insight (-0.5s to 0.0s):")
    scan_and_print_discrete_nodes(theta_frontal, 'Theta', 'Pre-Insight Frontal Theta', 'Pre')
    print(" POST-Insight (0.0s to 0.45s):")
    scan_and_print_discrete_nodes(theta_frontal, 'Theta', 'Post-Insight Frontal Theta', 'Post')
    
    print("\n--- 2. Right Parieto-Occipital Alpha (Visual / Brainstorming) ---")
    print(" PRE-Insight (-0.5s to 0.0s):")
    scan_and_print_discrete_nodes(alpha_rpo, 'Alpha', 'Pre-Insight R.P.O. Alpha', 'Pre')
    print(" POST-Insight (0.0s to 0.45s):")
    scan_and_print_discrete_nodes(alpha_rpo, 'Alpha', 'Post-Insight R.P.O. Alpha', 'Post')
    
    print("\n--- 3. Right Superior Temporal Gamma (The Aha! Spark) ---")
    print(" POST-Insight (0.0s to 0.45s):")
    scan_and_print_discrete_nodes(gamma_rst, 'RSTGamma', 'Post-Insight R.S.T. Gamma', 'Post')

    # -------------------------------------------------------------------------
    # ANALYSIS 9: Median Split High-DAT vs Low-DAT Aha! Burst Response
    # -------------------------------------------------------------------------
    print(f"\n=======================================================")
    print(f"[Phase 9] MEDIAN-SPLIT High-DAT vs Low-DAT Aha! Burst Arrays")
    print(f"=======================================================")
    
    dat_median = df_final['DAT_Score'].median()
    df_high = df_final[df_final['DAT_Score'] >= dat_median]
    df_low = df_final[df_final['DAT_Score'] < dat_median]
    print(f"  -> High DAT Group (N={len(df_high)}): DAT >= {dat_median}")
    print(f"  -> Low DAT Group  (N={len(df_low)}): DAT < {dat_median}")
    
    for metric_col, metric_name in [('Gamma', 'Right Temporal Gamma Burst'), ('PrefrontalGamma', 'Prefrontal Gamma Burst'), ('Alpha', 'Frontal Alpha Offloading')]:
        print(f"\n  --- Testing {metric_name} ---")
        t_high, p_high = stats.ttest_rel(df_high[f'{metric_col}_Treatment'], df_high[f'{metric_col}_Control'])
        t_low, p_low = stats.ttest_rel(df_low[f'{metric_col}_Treatment'], df_low[f'{metric_col}_Control'])
        
        print(f"     => High-DAT Cohort LLM Shift: t={t_high:6.3f}, p={p_high:.4f} " + ("<<< SIGNIFICANT!" if p_high < 0.05 else ""))
        print(f"     => Low-DAT  Cohort LLM Shift: t={t_low:6.3f}, p={p_low:.4f} " + ("<<< SIGNIFICANT!" if p_low < 0.05 else ""))

    # -------------------------------------------------------------------------
    # ANALYSIS 10: AUT Sub-component Modulators on Delta Shifts
    # -------------------------------------------------------------------------
    print(f"\n=======================================================")
    print(f"[Phase 10] AUT SUB-COMPONENT HEATMAP PREDICTORS (Delta LLM-Alone)")
    print(f"=======================================================")
    
    df_final['PrefrontalGamma_Diff'] = df_final['PrefrontalGamma_Treatment'] - df_final['PrefrontalGamma_Control']
    
    aut_cols = ['Fluency', 'Flexibility', 'Elaboration', 'Originality']
    erp_metrics = [
        ('Aha! Spark (RST Gamma)', 'Gamma_Diff'),
        ('Executive Insight (Prefrontal Gamma)', 'PrefrontalGamma_Diff'),
        ('Internal Ideation (Frontal Alpha)', 'Alpha_Diff')
    ]
    
    found_any_subscore = False
    for metric_name, diff_col in erp_metrics:
        print(f"\n  Top Predictors for {metric_name} Amplitude Shift:")
        for aut_sub in aut_cols:
            r, p_val = stats.pearsonr(df_final[aut_sub], df_final[diff_col])
            if p_val < 0.05:
                print(f"     * {aut_sub} heavily predicts the LLM-induced shift (r={r:.3f}, p={p_val:.4f})")
                found_any_subscore = True
                
                plt.figure(figsize=(7, 5))
                sns.regplot(data=df_final, x=aut_sub, y=diff_col, scatter_kws={'s':60, 'alpha':0.7}, line_kws={'color':'red'})
                plt.title(f"ERP Predictor: {aut_sub} vs {metric_name} Shift\nPearson r={r:.3f}, p={p_val:.4f}", fontweight='bold')
                plt.xlabel(f"AUT Sub-Score: {aut_sub}")
                plt.ylabel(f"LLM Shift Magnitude ({diff_col})")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"Scatter_ERP_Subscore_{aut_sub}_vs_{diff_col}.png"), dpi=600)
                plt.close()
                
        if not found_any_subscore: print("     -> No significant trait predictors found for this metric.")
        found_any_subscore = False

    print(f"\nHighly Detailed Statistical Visualizations exported natively to {output_dir}/")

if __name__ == "__main__":
    main()
