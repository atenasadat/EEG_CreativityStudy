import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import coherence
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mne
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "data_dir": "/Users/athenasaghi/Desktop/CleanDATA/clean/",
    "behavior_path": "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/",
    "freq_bands": {"Alpha": (8, 13), "Gamma": (35, 50)},
    "channels": ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1', 'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'P8', 'T8', 'Pz']
}

# ==========================================
# GRAPH METRIC EXTRACTION
# ==========================================
def extract_topology_features(raw, fs, bands, channels):
    """
    Computes global graph theoretic features for a raw EEG session.
    Features: Global Efficiency, Clustering Coefficient, Path Length.
    """
    data = raw.get_data(picks=channels)
    n_chans = len(channels)
    features = {}
    
    for band_name, (fmin, fmax) in bands.items():
        # Compute Coherence Matrix
        adj_matrix = np.zeros((n_chans, n_chans))
        for i in range(n_chans):
            for j in range(i+1, n_chans):
                f, coh = coherence(data[i], data[j], fs=fs, nperseg=1024)
                mask = (f >= fmin) & (f <= fmax)
                adj_matrix[i, j] = adj_matrix[j, i] = np.mean(coh[mask])
        
        # Threshold to create Graph
        threshold = np.median(adj_matrix)
        G = nx.from_numpy_array((adj_matrix > threshold).astype(int))
        
        # Topograhical Metrics
        features[f"{band_name}_AvgClustering"] = nx.average_clustering(G)
        features[f"{band_name}_GlobalEfficiency"] = nx.global_efficiency(G)
        try:
            features[f"{band_name}_AvgPathLength"] = nx.average_shortest_path_length(G)
        except nx.NetworkXNoPath:
            features[f"{band_name}_AvgPathLength"] = 0
            
    return features

# ==========================================
# MAIN ANALYSIS PIPELINE
# ==========================================
def run_synergy_prediction():
    print("Starting RQ4 Neural Topology Analysis...")
    
    # 1. Load Data
    dat_df = pd.read_csv(os.path.join(CONFIG["behavior_path"], "DATScores.csv"))
    aut_df = pd.read_csv(os.path.join(CONFIG["behavior_path"], "AUTScores.csv"))
    dat_df['Pid'] = dat_df['Pid'].astype(str).str.replace(' ', '')
    aut_df['Pid'] = aut_df['Pid'].astype(str).str.replace(' ', '')
    beh_df = pd.merge(dat_df, aut_df, on='Pid')

    # 2. Extract Features (Subject Level)
    rows = []
    files = [f for f in os.listdir(CONFIG["data_dir"]) if f.endswith('.set') and 'control' in f.lower()]
    
    # Progress bar for feature extraction
    for filename in tqdm(files[:10], desc="Extracting Topology Features"): 
        pid = filename.split('_')[0]
        if pid not in beh_df['Pid'].values: continue
        
        raw = mne.io.read_raw_eeglab(os.path.join(CONFIG["data_dir"], filename), preload=True, verbose=False)
        raw.pick(CONFIG["channels"])
        
        feats = extract_topology_features(raw, raw.info['sfreq'], CONFIG["freq_bands"], CONFIG["channels"])
        feats['Pid'] = pid
        feats['Target_DAT'] = beh_df[beh_df['Pid'] == pid]['dat'].values[0]
        rows.append(feats)
        
    df_features = pd.DataFrame(rows)
    
    # 3. Predict AI-Synergy (Using DAT as proxy for baseline divergent trait)
    if len(df_features) < 3:
        print("Warning: Insufficient data for regression. Need more matched participants.")
        return

    X = df_features.drop(columns=['Pid', 'Target_DAT'])
    y = df_features['Target_DAT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nPrediction Results (DAT Target):")
    print(f"  -> Model MSE: {mse:.4f}")
    print(f"  -> Feature Importances: {dict(zip(X.columns, model.feature_importances_))}")

if __name__ == "__main__":
    run_synergy_prediction()
