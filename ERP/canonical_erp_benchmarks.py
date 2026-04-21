import os
import numpy as np
import pandas as pd
import mne
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress MNE spatial warnings native to EEGLAB channel mapping
warnings.filterwarnings("ignore")

def get_mean_amplitude(epochs, channels, tmin, tmax):
    """
    Computes pure voltage amplitude across an exact millisecond window.
    Strictly utilized for true Event-Related Potentials (N400, P300, LPC).
    """
    evoked = epochs.average().pick(channels)
    evoked.crop(tmin, tmax)
    # Return as microvolts (uV) to maintain standard EEG literature aesthetics
    return np.mean(evoked.data) * 1e6

def get_mean_tfr(epochs, channels, tmin, tmax, fmin, fmax):
    """
    Computes absolute oscillatory power via the continuous Morlet Wavelet.
    Strictly utilized for non-phase-locked induced bands (Gamma Sparks, Alpha Sync).
    """
    epochs_roi = epochs.copy().pick(channels)
    freqs = np.arange(fmin, fmax + 1, 2.0)
    # Using dynamic cycle ratios natively tuned to the frequency bands
    tfr = epochs_roi.compute_tfr(method="morlet", freqs=freqs, n_cycles=freqs/3.0, return_itc=False, average=True, verbose=False)
    tfr.crop(tmin, tmax)
    return np.mean(tfr.data)

def generate_paired_plot(df_focused, metric_name, t, p, title, out_name, output_dir):
    """
    Generates a high-fidelity academic paired violin plot tracing individual 
    subject variances longitudinally across the two task states.
    """
    plt.figure(figsize=(6, 6))
    df_melt = df_focused[['Pid', f'{metric_name}_Control', f'{metric_name}_Treatment']].melt(id_vars='Pid', var_name='Condition', value_name='Amplitude')
    df_melt['Condition'] = df_melt['Condition'].map({f'{metric_name}_Control': 'Unassisted (Alone)', f'{metric_name}_Treatment': 'LLM Assisted'})
    
    sns.violinplot(data=df_melt, x='Condition', y='Amplitude', hue='Condition', palette={'Unassisted (Alone)': '#D25353', 'LLM Assisted': '#5A9CB5'}, legend=False)
    sns.stripplot(data=df_melt, x='Condition', y='Amplitude', color='black', alpha=0.5, size=6, jitter=False)
    
    pivoted = df_melt.pivot(index='Pid', columns='Condition', values='Amplitude')
    for i in range(len(pivoted)):
        plt.plot([0, 1], [pivoted.iloc[i]['Unassisted (Alone)'], pivoted.iloc[i]['LLM Assisted']], color='gray', alpha=0.3, zorder=0)
        
    plt.title(f"{title}\nPaired t-test: t = {t:.3f}, p = {p:.4f}", fontweight='bold')
    
    # Adjust Y-axis label dynamically based on whether it is a True Voltage ERP or an Oscillatory Map
    if metric_name in ['N400', 'P300', 'LPC']:
        plt.ylabel("Voltage Amplitude (μV)")
    else:
        plt.ylabel(r"Absolute Oscillatory Power ($\mu V^2$)")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{out_name}.png"), dpi=600)
    plt.close()

def main():
    base_path = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    output_dir = "canonical_erp_stats"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Ingest Psychometric Behavior Metrics
    project_root = "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/"
    dat_df = pd.read_csv(os.path.join(project_root, 'DATScores.csv'))
    aut_df = pd.read_csv(os.path.join(project_root, 'AUTScores.csv'))
    df_beh = pd.merge(dat_df[['Pid', 'dat']], aut_df[['Pid', 'Fluency', 'Flexibility', 'Elaboration', 'Originality']], on='Pid', how='inner')
    df_beh.rename(columns={'dat': 'DAT_Score'}, inplace=True)
    
    c_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'control' in f.lower()]
    t_files = [f for f in os.listdir(base_path) if f.endswith('.set') and 'postcleaning' in f and 'treatment' in f.lower()]
    c_dict = {f.split('_')[0]: f for f in c_files}
    t_dict = {f.split('_')[0]: f for f in t_files}
    matched_pids = sorted(list(set(c_dict.keys()) & set(t_dict.keys())), key=lambda x: int(x.replace('P', '')))
    
    print("\n=======================================================")
    print("CANONICAL ERP & TIME-FREQUENCY BENCHMARK SCRIPT")
    print("=======================================================")
    print(f"[SYSTEM] Extracting Topographical Voltage arrays for N={len(matched_pids)}...\n")
    
    results = []
    
    for pid in matched_pids:
        # Load Control Parameters
        raw_c = mne.io.read_raw_eeglab(os.path.join(base_path, c_dict[pid]), preload=True, verbose=False)
        events_c, event_id_c = mne.events_from_annotations(raw_c, verbose=False)
        # Epoch expanded to +1.0 seconds intentionally to trace the Late Positive Component explicitly
        ep_c = mne.Epochs(raw_c, events_c, event_id_c.get('100', None), tmin=-1.0, tmax=1.0, baseline=(-0.2, 0), preload=True, verbose=False) if '100' in event_id_c else None
        
        # Load Treatment (LLM) Parameters
        raw_t = mne.io.read_raw_eeglab(os.path.join(base_path, t_dict[pid]), preload=True, verbose=False)
        events_t, event_id_t = mne.events_from_annotations(raw_t, verbose=False)
        ep_t = mne.Epochs(raw_t, events_t, event_id_t.get('100', None), tmin=-1.0, tmax=1.0, baseline=(-0.2, 0), preload=True, verbose=False) if '100' in event_id_t else None
        
        if ep_c is None or ep_t is None: 
            print(f"Skipping {pid} due to missing anchor triggers.")
            continue
        
        # -------------------------------------------------------------------------
        # Marker 1: The Kounios & Beeman Insight Gamma Spark
        # Core Theory: Right Temporal / Right Parietal power bursts immediately pre-insight
        # -------------------------------------------------------------------------
        g_c = get_mean_tfr(ep_c, ['T8', 'P8'], -0.3, 0.0, 35, 50)
        g_t = get_mean_tfr(ep_t, ['T8', 'P8'], -0.3, 0.0, 35, 50)
        
        # -------------------------------------------------------------------------
        # Marker 2: The Fink Prefrontal Alpha Synchronization (Task-Related Ideation)
        # Core Theory: Prefrontal cortices lock into synchronized alpha directly preceding internal idea generation.
        # -------------------------------------------------------------------------
        a_c = get_mean_tfr(ep_c, ['Fp1', 'Fp2', 'Fz'], -0.5, 0.0, 8, 13)
        a_t = get_mean_tfr(ep_t, ['Fp1', 'Fp2', 'Fz'], -0.5, 0.0, 8, 13)
        
        # -------------------------------------------------------------------------
        # Marker 3: The P300 / P3b (Attentional Resource Allocation)
        # Core Theory: P3b amplitude suppression represents "defocused attention", opening the bottleneck to novel logic.
        # -------------------------------------------------------------------------
        p3_c = get_mean_amplitude(ep_c, ['Pz', 'P3', 'P4'], 0.25, 0.45)
        p3_t = get_mean_amplitude(ep_t, ['Pz', 'P3', 'P4'], 0.25, 0.45)
        
        # -------------------------------------------------------------------------
        # Marker 4: The N400 (Semantic Divergence / Latent Incongruity Mapping)
        # Core Theory: Steeper negative amplitude signifies harsh rejection of semantically weird inputs.
        # -------------------------------------------------------------------------
        n4_c = get_mean_amplitude(ep_c, ['Cz', 'Pz'], 0.3, 0.5)
        n4_t = get_mean_amplitude(ep_t, ['Cz', 'Pz'], 0.3, 0.5)
        
        # -------------------------------------------------------------------------
        # Marker 5: The Late Positive Component (LPC - Conscious Evaluation)
        # Core Theory: The LPC activates exclusively when you must *judge* whether an idea is good or bad.
        # -------------------------------------------------------------------------
        lpc_c = get_mean_amplitude(ep_c, ['Cz', 'Pz'], 0.50, 0.80)
        lpc_t = get_mean_amplitude(ep_t, ['Cz', 'Pz'], 0.50, 0.80)
        
        results.append({
            'Pid': pid,
            'InsightGamma_Control': g_c, 'InsightGamma_Treatment': g_t,
            'FrontalAlpha_Control': a_c, 'FrontalAlpha_Treatment': a_t,
            'P300_Control': p3_c, 'P300_Treatment': p3_t,
            'N400_Control': n4_c, 'N400_Treatment': n4_t,
            'LPC_Control': lpc_c, 'LPC_Treatment': lpc_t
        })
        
    df_eeg = pd.DataFrame(results)
    df = pd.merge(df_beh, df_eeg, on='Pid', how='inner')
    
    benchmarks = [
        ('InsightGamma', 'Kounios & Beeman Insight Gamma Burst\n(-0.3s to 0s) Right Temporal'),
        ('FrontalAlpha', 'Fink Frontal Alpha Synchronization\n(-0.5s to 0s) Prefrontal'),
        ('P300', 'P300 Attentional Defocusing Array\n(+0.25s to +0.45s) Parietal Voltage'),
        ('N400', 'N400 Semantic Divergence Integration\n(+0.3s to +0.5s) Centro-Parietal Voltage'),
        ('LPC', 'Late Positive Component (Conscious Evaluation)\n(+0.5s to +0.8s) Centro-Parietal Voltage')
    ]
    
    print("===============================================")
    print("CANONICAL BIOMARKER T-TEST RESULTS")
    print("===============================================\n")
    
    for metric_col, title in benchmarks:
        t_stat, p_val = stats.ttest_rel(df[f'{metric_col}_Treatment'], df[f'{metric_col}_Control'])
        
        marker = "<<< SIGNIFICANT" if p_val < 0.05 else ("< Trending" if p_val < 0.1 else "")
        print(f"[{metric_col}] Shift Vector:")
        print(f"   -> t-stat = {t_stat:6.3f}, p = {p_val:.4f} {marker}\n")
        
        generate_paired_plot(df, metric_col, t_stat, p_val, title, f"Canonical_{metric_col}", output_dir)
        
    print("===============================================")
    print("PSYCHOMETRIC TRAIT PREDICTORS (Delta Shifts)")
    print("===============================================\n")
    
    traits = ['DAT_Score', 'Fluency', 'Flexibility', 'Elaboration', 'Originality']
    
    for metric_col, title in benchmarks:
        print(f"[{metric_col}] Trait Modulators:")
        df[f'{metric_col}_Diff'] = df[f'{metric_col}_Treatment'] - df[f'{metric_col}_Control']
        
        found_any = False
        for trait in traits:
            r, p_val = stats.pearsonr(df[trait], df[f'{metric_col}_Diff'])
            if p_val < 0.05:
                print(f"   * {trait} predicts shift magnitude! (r={r:.3f}, p={p_val:.4f})")
                found_any = True
                
                plt.figure(figsize=(7, 5))
                sns.regplot(data=df, x=trait, y=f'{metric_col}_Diff', color='#724d9c', scatter_kws={'s': 80})
                plt.axhline(0, color='gray', linestyle='--')
                plt.title(f"Canonical Predictor: {trait} vs {metric_col} Shift\nPearson's r = {r:.3f}, p-val = {p_val:.4f}", fontweight='bold')
                plt.xlabel(f"Psychometric Trait ({trait})")
                plt.ylabel(f"{metric_col} Shift Magnitude")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"Predictor_Canonical_{metric_col}_vs_{trait}.png"), dpi=600)
                plt.close()
                
        if not found_any: print("   -> No significant psychological trait predictors.")
        print("")
        
    print(f"[SUCCESS] Violin and Psychometric scatter topologies exported to {output_dir}/")

if __name__ == "__main__":
    main()
