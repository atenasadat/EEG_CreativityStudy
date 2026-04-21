import os
import glob
import numpy as np
import pandas as pd
import mne
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Disable MNE warnings globally for a clean ML output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

class EEGCreativityDataset(Dataset):
    """
    Standardize Dataset for the EEG Creativity Study.
    Handles windows of EEG data and associated psychometric scores.
    """
    def __init__(self, data_path, behavior_path, window_size=500, stride=250, transform=None):
        """
        Args:
            data_path (str): Path to directory containing .set files.
            behavior_path (str): Path to directory containing DAT/AUT CSVs.
            window_size (int): Number of samples per window.
            stride (int): Hop size between windows.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        # Standardize to 19 channels (common across study)
        self.ch_names = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1', 'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'P8', 'T8', 'Pz']
        
        # Silence verbose MNE output
        mne.set_log_level('ERROR')
        
        # Load behavior data
        self.dat_df = pd.read_csv(os.path.join(behavior_path, "DATScores.csv"))
        self.aut_df = pd.read_csv(os.path.join(behavior_path, "AUTScores.csv"))
        self.dat_df['Pid'] = self.dat_df['Pid'].astype(str).str.replace(' ', '')
        self.aut_df['Pid'] = self.aut_df['Pid'].astype(str).str.replace(' ', '')
        
        self.behavior_df = pd.merge(self.dat_df, self.aut_df, on='Pid')
        
        # Match files
        self.files = [f for f in os.listdir(data_path) if f.endswith('.set') and 'postcleaning' in f]
        self.samples = []
        self.data_cache = {} # Dictionary to store in-memory raw arrays
        self._prepare_samples()

    def _prepare_samples(self):
        """Pre-index windows and load all subject data into RAM for near-instant access."""
        from tqdm import tqdm
        print(f"\n[LOADER] Caching {len(self.files)} subject files into RAM for high-speed training...")
        
        for filename in tqdm(self.files, desc="Loading EEG Data"):
            pid = filename.split('_')[0]
            condition = 'treatment' if 'treatment' in filename.lower() else 'control'
            
            # Check if we have behavior data for this PID
            if pid not in self.behavior_df['Pid'].values:
                continue
                
            beh_row = self.behavior_df[self.behavior_df['Pid'] == pid].iloc[0]
            
            try:
                # Load the full file into memory once
                raw = mne.io.read_raw_eeglab(os.path.join(self.data_path, filename), preload=True, verbose=False)
                raw.pick(self.ch_names)
                
                # Store the underlying numpy array
                self.data_cache[filename] = raw.get_data()
                
                n_samples = self.data_cache[filename].shape[1]
                
                # Create window indices
                for start in range(0, n_samples - self.window_size, self.stride):
                    self.samples.append({
                        'file': filename,
                        'start': start,
                        'pid': pid,
                        'condition': 1 if condition == 'treatment' else 0,
                        'dat': beh_row['dat'],
                        'aut_total': beh_row['Total']
                    })
            except Exception as e:
                print(f"\n   [!] Error indexing {filename}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        
        # High-speed in-memory slice (O(1) complexity)
        data = self.data_cache[sample_meta['file']][:, sample_meta['start']:sample_meta['start'] + self.window_size]
        
        # Standard normalization (Z-score per window)
        # We handle this here to ensure models receive stable inputs
        data = (data - np.mean(data)) / (np.std(data) + 1e-9)
        
        x = torch.tensor(data, dtype=torch.float32)
        y_cond = torch.tensor(sample_meta['condition'], dtype=torch.long)
        y_dat = torch.tensor(sample_meta['dat'], dtype=torch.float32)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y_cond, y_dat, sample_meta['pid']

def get_dataloader(data_path, behavior_path, batch_size=32, shuffle=True):
    dataset = EEGCreativityDataset(data_path, behavior_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test block
    print("Testing Data Loader Infrastructure...")
    # These paths would need to be adjusted to the user's actual local data
    DATA_DIR = "/Users/athenasaghi/Desktop/CleanDATA/clean/"
    BEH_DIR = "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/"
    
    if os.path.exists(DATA_DIR):
        loader = get_dataloader(DATA_DIR, BEH_DIR, batch_size=4)
        for x, cond, dat, pids in loader:
            print(f"Batch X shape: {x.shape}") # [Batch, Channels, Time]
            print(f"Condition labels: {cond}")
            print(f"DAT scores: {dat}")
            break
    else:
        print(f"Data directory {DATA_DIR} not found. Skipping live test.")
