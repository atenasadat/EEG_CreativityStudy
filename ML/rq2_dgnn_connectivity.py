import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from utils.data_loader import EEGCreativityDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "num_features": 1, # Using electrode voltage at time t as feature
    "num_classes": 2,  # Alone vs Assisted
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 30,
    "data_dir": "/Users/athenasaghi/Desktop/CleanDATA/clean/",
    "beh_dir": "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# MODEL DEFINITION (ST-GCN)
# ==========================================
class SpatioTemporalGNN(nn.Module):
    """
    Graph Neural Network for analyzing brain connectivity dynamics.
    Utilizes Graph Convolutions to capture spatial dependencies and 
    GRU to capture temporal evolution of the neural "decoupling" signal.
    """
    def __init__(self, num_nodes=19, num_features=1, hidden_dim=64):
        super(SpatioTemporalGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, CONFIG["num_classes"])

    def forward(self, x, edge_index, batch_idx):
        """
        Args:
            x: [Batch * Nodes, Time, Features]
            edge_index: Graph connectivity
            batch_idx: Mapping for global pooling
        """
        # Collapse time for spatial processing or process per step
        batch_size = batch_idx.max().item() + 1
        num_nodes = x.shape[0] // batch_size
        time_steps = x.shape[1]
        
        # We'll extract spatial features at each time step (downsampled)
        temporal_features = []
        for t in range(0, time_steps, 50): # Downsample for efficiency
            x_t = x[:, t, :]
            h = self.conv1(x_t, edge_index)
            h = F.relu(h)
            h = self.conv2(h, edge_index)
            h = global_mean_pool(h, batch_idx) # [Batch, Hidden]
            temporal_features.append(h.unsqueeze(1))
            
        temporal_features = torch.cat(temporal_features, dim=1) # [Batch, Time, Hidden]
        _, h_n = self.rnn(temporal_features)
        out = self.fc(h_n.squeeze(0))
        return out

# ==========================================
# GRAPH UTILITIES
# ==========================================
def create_static_edge_index(num_nodes=19):
    """Creates a fully connected graph for baseline analysis."""
    adj = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(adj, 0)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    return edge_index

# ==========================================
# TRAINING PIPELINE
# ==========================================
def train():
    device = torch.device(CONFIG["device"])
    dataset = EEGCreativityDataset(CONFIG["data_dir"], CONFIG["beh_dir"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    edge_index = create_static_edge_index().to(device)
    model = SpatioTemporalGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting RQ2 Dynamic GNN Training...")
    
    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        correct = 0
        # Add tqdm for batch progress
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for x, cond, _, _ in pbar:
            # Prepare PyG format
            B, C, T = x.shape
            x = x.permute(0, 1, 2).reshape(B*C, T, 1).to(device) # [B*Nodes, T, Feat]
            cond = cond.to(device)
            batch_mapping = torch.arange(B).repeat_interleave(C).to(device)
            
            optimizer.zero_grad()
            out = model(x, edge_index, batch_mapping)
            loss = criterion(out, cond)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == cond).sum().item()
            
            pbar.set_postfix({'loss': total_loss / (correct/B + 1e-6), 'acc': correct / (B * len(pbar))})
            
        acc = correct / len(dataset)
        if (epoch + 1) % 5 == 0:
            print(f" -> Final Epoch Acc: {acc:.4f}")

if __name__ == "__main__":
    try:
        import torch_geometric
        train()
    except ImportError:
        print("Error: torch_geometric not found. Please install via requirements.txt.")
