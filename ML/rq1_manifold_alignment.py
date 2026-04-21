import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.data_loader import EEGCreativityDataset
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "latent_dim": 2,
    "input_channels": 19,
    "sequence_length": 500,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 50,
    "data_dir": "/Users/athenasaghi/Desktop/CleanDATA/clean/",
    "beh_dir": "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# MODEL DEFINITION (VAE)
# ==========================================
class EEGVAE(nn.Module):
    """
    Variational Autoencoder optimized for EEG window manifold learning.
    Projects high-dimensional neural oscillations into a 2D creative manifold.
    """
    def __init__(self, channels, seq_len, latent_dim):
        super(EEGVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (seq_len // 4), 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * (seq_len // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (64, seq_len // 4)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Assumes normalized input [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z = self.decoder_input(z)
        return self.decoder(z), mu, logvar

# ==========================================
# TRAINING LOGIC
# ==========================================
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train():
    device = torch.device(CONFIG["device"])
    
    # Initialize Dataset
    if not os.path.exists(CONFIG["data_dir"]):
        print(f"Error: Data directory {CONFIG['data_dir']} not found. Please update CONFIG.")
        return

    dataset = EEGCreativityDataset(CONFIG["data_dir"], CONFIG["beh_dir"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    model = EEGVAE(CONFIG["input_channels"], CONFIG["sequence_length"], CONFIG["latent_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    
    print(f"Starting RQ1 Manifold Training on {device}...")
    
    model.train()
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0
        # Use tqdm for batch progress
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch_idx, (data, cond, _, _) in enumerate(pbar):
            data = data.to(device)
            # Normalize for VAE (approximation)
            data = (data - data.mean()) / (data.std() + 1e-6)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f" -> Final Avg Loss: {avg_loss:.4f}")

    # Visualization
    visualize_manifold(model, dataloader, device)

def visualize_manifold(model, dataloader, device):
    model.eval()
    latents = []
    labels = []
    
    print("Generating Manifold Visualization...")
    with torch.no_grad():
        for data, cond, _, _ in dataloader:
            data = data.to(device)
            _, mu, _ = model(data)
            latents.append(mu.cpu().numpy())
            labels.append(cond.numpy())
            if len(latents) > 20: break # Sample for speed

    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Condition (0=Alone, 1=AI Assisted)')
    plt.title("EEG Creative State Manifold (VAE Latent Space)")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.grid(True, alpha=0.3)
    plt.savefig("ML/rq1_manifold_result.png")
    print("Visualization saved to ML/rq1_manifold_result.png")

if __name__ == "__main__":
    train()
