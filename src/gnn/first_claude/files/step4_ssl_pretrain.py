"""
STEP 4: SSL PRE-TRAINING (Control vs Epilepsy)
===============================================
Self-supervised pre-training using contrastive learning.

Method: SimCLR-style with graph augmentations
- Edge dropout
- Feature masking  
- Edge noise

This step is UNSUPERVISED - it doesn't use labels!
The goal is to learn good EEG representations that will help with classification.

Output:
- pretrained_encoder.pt: Pre-trained GNN encoder
- ssl_training_loss.png: Training curve
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

GRAPHS_FILE = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\control_vs_epilepsy\pyg_dataset\control_epilepsy_graphs.pt")
OUTPUT_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\control_vs_epilepsy\ssl_pretrained")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
TEMPERATURE = 0.5
HIDDEN_DIM = 64
PROJECTION_DIM = 64

# Augmentation parameters
EDGE_DROP_RATE = 0.3
FEATURE_MASK_RATE = 0.3
EDGE_NOISE_STD = 0.15
DROPOUT = 0.5

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# =============================================================================
# AUGMENTATION
# =============================================================================

def augment_graph(data, edge_drop=0.2, feat_mask=0.2, edge_noise=0.1):
    """Apply random augmentations to a graph."""
    # Clone data
    x = data.x.clone()
    edge_index = data.edge_index.clone()
    edge_attr = data.edge_attr.clone() if data.edge_attr is not None else None
    
    # 1. Edge Dropout
    if edge_drop > 0 and edge_index.shape[1] > 0:
        mask = torch.rand(edge_index.shape[1]) > edge_drop
        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]
    
    # 2. Feature Masking
    if feat_mask > 0:
        mask = torch.rand_like(x) > feat_mask
        x = x * mask.float()
    
    # 3. Edge Noise
    if edge_noise > 0 and edge_attr is not None:
        noise = torch.randn_like(edge_attr) * edge_noise
        edge_attr = torch.clamp(edge_attr + noise, 0, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =============================================================================
# MODEL
# =============================================================================

class GNNEncoder(nn.Module):
    """2-layer GAT encoder."""
    
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
    
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    """Full contrastive model: Encoder + Projection Head."""
    
    def __init__(self, in_channels, hidden_channels, projection_dim):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels)
        self.projector = ProjectionHead(hidden_channels, hidden_channels, projection_dim)
    
    def forward(self, x, edge_index, batch=None):
        h = self.encoder(x, edge_index, batch)
        z = self.projector(h)
        return h, z


# =============================================================================
# CONTRASTIVE LOSS
# =============================================================================

def contrastive_loss(z1, z2, temperature=0.5):
    """NT-Xent (InfoNCE) loss for contrastive learning."""
    batch_size = z1.shape[0]
    
    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate
    z = torch.cat([z1, z2], dim=0)
    
    # Similarity matrix
    sim = torch.mm(z, z.t()) / temperature
    
    # Labels: positive pairs
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ]).to(z.device)
    
    # Mask diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, float('-inf'))
    
    return F.cross_entropy(sim, labels)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, graphs, optimizer, batch_size=64):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    # Shuffle
    indices = np.random.permutation(len(graphs))
    
    for start in range(0, len(graphs), batch_size):
        end = min(start + batch_size, len(graphs))
        batch_idx = indices[start:end]
        
        if len(batch_idx) < 2:
            continue
        
        # Get batch
        batch_graphs = [graphs[i] for i in batch_idx]
        
        # Augment twice
        aug1 = [augment_graph(g, EDGE_DROP_RATE, FEATURE_MASK_RATE, EDGE_NOISE_STD) for g in batch_graphs]
        aug2 = [augment_graph(g, EDGE_DROP_RATE, FEATURE_MASK_RATE, EDGE_NOISE_STD) for g in batch_graphs]
        
        # Batch
        batch1 = Batch.from_data_list(aug1).to(DEVICE)
        batch2 = Batch.from_data_list(aug2).to(DEVICE)
        
        # Forward
        _, z1 = model(batch1.x, batch1.edge_index, batch1.batch)
        _, z2 = model(batch2.x, batch2.edge_index, batch2.batch)
        
        # Loss
        loss = contrastive_loss(z1, z2, TEMPERATURE)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STEP 4: SSL PRE-TRAINING (Control vs Epilepsy)")
    print("="*70)
    
    # Load graphs
    print(f"\nLoading graphs from: {GRAPHS_FILE}")
    graphs = torch.load(GRAPHS_FILE, weights_only=False)
    print(f"Loaded {len(graphs)} graphs")
    
    in_channels = graphs[0].x.shape[1]
    print(f"Input features: {in_channels}")
    
    # Create model
    model = ContrastiveModel(
        in_channels=in_channels,
        hidden_channels=HIDDEN_DIM,
        projection_dim=PROJECTION_DIM
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training
    print(f"\n{'='*70}")
    print(f"Training for {EPOCHS} epochs...")
    print(f"{'='*70}")
    
    losses = []
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, graphs, optimizer, BATCH_SIZE)
        losses.append(loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f}")
        
        # Save best
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'loss': loss
            }, OUTPUT_DIR / 'best_ssl_model.pt')
    
    # Save final encoder
    torch.save(model.encoder.state_dict(), OUTPUT_DIR / 'pretrained_encoder.pt')
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.title('SSL Pre-training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'ssl_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"✅ SSL PRE-TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"  - pretrained_encoder.pt")
    print(f"  - ssl_training_loss.png")
    print("\nNext step: Run step5_train_classifier.py")
