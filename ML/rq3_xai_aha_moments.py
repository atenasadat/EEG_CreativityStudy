import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.data_loader import EEGCreativityDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "batch_size": 32,
    "lr": 0.001,
    "epochs": 40,
    "data_dir": "/Users/athenasaghi/Desktop/CleanDATA/clean/",
    "beh_dir": "/Users/athenasaghi/VSProjects/EEG_CreativityStudy/",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==========================================
# MODEL DEFINITION (EEGNet)
# ==========================================
class EEGNet(nn.Module):
    """
    Standard EEGNet architecture (Lawhern et al., 2018).
    Compact CNN optimized for brain-computer interface and neural signal classification.
    """
    def __init__(self, num_channels=19, num_samples=500, num_classes=2):
        super(EEGNet, self).__init__()
        
        # Block 1 - Temporal and Spatial Convolutions
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        
        # Depthwise Conv to capture spatial relationships
        self.depthwise = nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 2 - Separable Convolution
        self.separable = nn.Conv2d(32, 32, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.25)
        
        # Classifier
        self.flatten = nn.Flatten()
        # Shape calculation for FC layer
        dummy_x = torch.zeros(1, 1, num_channels, num_samples)
        x = self.dropout1(self.pool1(self.elu(self.batchnorm2(self.depthwise(self.batchnorm1(self.conv1(dummy_x)))))))
        x = self.dropout2(self.pool2(self.elu(self.batchnorm3(self.separable(x)))))
        self.fc = nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], num_classes)

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1) # Add channel dimension [Batch, 1, Channels, Time]
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.separable(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        return self.fc(x)

# ==========================================
# TRAINING & XAI LOGIC
# ==========================================
def train_and_explain():
    device = torch.device(CONFIG["device"])
    dataset = EEGCreativityDataset(CONFIG["data_dir"], CONFIG["beh_dir"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    model = EEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    
    print("Starting RQ3 EEGNet training for 'Aha!' Moments...")
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        total_loss = 0
        for data, cond, _, _ in pbar:
            data, cond = data.to(device), cond.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, cond)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (len(pbar))})
            
        if (epoch + 1) % 10 == 0:
            print(f" -> Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # Apply Captum Integrated Gradients
    try:
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(model)
        
        # Take a sample batch for explanation
        sample_input, _, _, _ = next(iter(dataloader))
        sample_input = sample_input[:1].to(device).requires_grad_()
        
        print("Calculating Neural Feature Attribution (XAI)...")
        attributions = ig.attribute(sample_input, target=1)
        
        # attributions size: [1, 1, Channels, Time]
        spatial_importance = attributions.abs().mean(dim=(0, 1, 3))
        print(f"Topograhical Importance Map (per channel): {spatial_importance.cpu().detach().numpy()}")
        
    except ImportError:
        print("Captum not found. Skipping XAI attribution phase.")

if __name__ == "__main__":
    train_and_explain()
