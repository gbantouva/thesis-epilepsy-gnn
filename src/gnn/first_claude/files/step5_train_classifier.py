"""
STEP 5: CLASSIFIER TRAINING (Control vs Epilepsy)
==================================================
Train a GNN classifier to distinguish control vs epilepsy patients.

Task: Binary classification
- Class 0: Control (healthy)
- Class 1: Epilepsy

Uses SSL pre-trained encoder + fine-tuning with strong regularization.

Output:
- best_classifier.pt: Best model checkpoint
- test_predictions.npz: Final test predictions
- Various plots: training curves, confusion matrix, etc.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\gnn\control_vs_epilepsy\pyg_dataset")
SSL_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\gnn\control_vs_epilepsy\ssl_pretrained")
OUTPUT_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\gnn\control_vs_epilepsy\classifier")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0005
HIDDEN_DIM = 64
NUM_CLASSES = 2
WEIGHT_DECAY = 1e-4
PATIENCE = 10
MIN_DELTA = 0.001

# Data augmentation
USE_AUGMENTATION = True
EDGE_DROP_RATE = 0.3
FEATURE_NOISE_STD = 0.05
FEATURE_MASK_RATE = 0.2

# Model settings
USE_PRETRAINED = True
FREEZE_ENCODER = False

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# =============================================================================
# MODEL
# =============================================================================

class GNNEncoder(nn.Module):
    """2-layer GAT encoder (same as SSL)."""
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)
        self.conv2 = GATConv(hidden_channels * 2, out_channels, heads=1, concat=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x


class EpilepsyClassifier(nn.Module):
    """Binary classifier: Control vs Epilepsy."""
    
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, hidden_channels)
        
        # Simple classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        h = self.encoder(x, edge_index, batch)
        out = self.classifier(h)
        return out


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_batch(batch, training=True):
    """Apply augmentation during training."""
    if not training or not USE_AUGMENTATION:
        return batch
    
    x = batch.x.clone()
    edge_index = batch.edge_index.clone()
    
    # Edge dropout
    if torch.rand(1).item() > 0.5 and edge_index.shape[1] > 0:
        edge_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > EDGE_DROP_RATE
        edge_index = edge_index[:, edge_mask]
    
    # Feature noise
    if torch.rand(1).item() > 0.5:
        noise = torch.randn_like(x) * FEATURE_NOISE_STD
        x = x + noise
    
    # Feature masking
    if torch.rand(1).item() > 0.5:
        feat_mask = torch.rand_like(x) > FEATURE_MASK_RATE
        x = x * feat_mask.float()
    
    # Rebuild batch
    from torch_geometric.data import Batch, Data
    augmented_data = []
    ptr = batch.ptr
    
    for i in range(len(ptr) - 1):
        start_idx = ptr[i]
        end_idx = ptr[i + 1]
        edge_mask = (batch.batch[edge_index[0]] == i)
        graph_edges = edge_index[:, edge_mask] - start_idx
        
        augmented_data.append(Data(
            x=x[start_idx:end_idx],
            edge_index=graph_edges,
            y=batch.y[i:i+1]
        ))
    
    return Batch.from_data_list(augmented_data)


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(DEVICE)
        batch = augment_batch(batch, training=True)
        
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in loader:
        batch = batch.to(DEVICE)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        total_loss += loss.item() * batch.num_graphs
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())
        all_probs.extend(probs)
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, acc, f1, auc, all_preds, all_labels, all_probs


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STEP 5: CLASSIFIER TRAINING (Control vs Epilepsy)")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    train_data = torch.load(DATA_DIR / 'train_graphs.pt', weights_only=False)
    val_data = torch.load(DATA_DIR / 'val_graphs.pt', weights_only=False)
    test_data = torch.load(DATA_DIR / 'test_graphs.pt', weights_only=False)
    
    print(f"  Train: {len(train_data)} graphs")
    print(f"  Val:   {len(val_data)} graphs")
    print(f"  Test:  {len(test_data)} graphs")
    
    # Create loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    in_channels = train_data[0].x.shape[1]
    model = EpilepsyClassifier(in_channels, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    
    # Load pre-trained encoder
    if USE_PRETRAINED:
        print("\nLoading SSL pre-trained encoder...")
        pretrained_path = SSL_DIR / 'pretrained_encoder.pt'
        if pretrained_path.exists():
            model.encoder.load_state_dict(torch.load(pretrained_path, weights_only=True))
            print("  ✓ Loaded pre-trained weights")
            
            if FREEZE_ENCODER:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                print("  ✓ Encoder frozen")
        else:
            print(f"  ⚠️  Pre-trained weights not found at {pretrained_path}")
            print("  Training from scratch...")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    # Optimizer & criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=3, verbose=True
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
        'train_val_gap': [], 'lr': []
    }
    
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_acc, val_f1, val_auc, _, _, _ = evaluate(model, val_loader)
        
        # Update scheduler
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['train_val_gap'].append(train_acc - val_acc)
        history['lr'].append(current_lr)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                  f"AUC: {val_auc:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_auc > best_val_auc + MIN_DELTA:
            best_val_auc = val_auc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'train_val_gap': train_acc - val_acc
            }, OUTPUT_DIR / 'best_classifier.pt')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break
    
    # =============================================================================
    # TEST EVALUATION
    # =============================================================================
    
    print(f"\n{'='*70}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*70}")
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / 'best_classifier.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nBest model from epoch: {checkpoint['epoch']}")
    print(f"  Val AUC: {checkpoint['val_auc']:.4f}")
    print(f"  Val Acc: {checkpoint['val_acc']:.4f}")
    print(f"  Val F1:  {checkpoint['val_f1']:.4f}")
    
    # Test
    test_loss, test_acc, test_f1, test_auc, test_preds, test_labels, test_probs = evaluate(model, test_loader)
    
    print(f"\n🎯 TEST RESULTS:")
    print(f"{'='*70}")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  AUC-ROC:   {test_auc:.4f}")
    print(f"{'='*70}")
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\nConfusion Matrix:")
    print(f"  {'':>12} Predicted")
    print(f"  {'':>12} Control  Epilepsy")
    print(f"  Actual Control  {cm[0,0]:>6}    {cm[0,1]:>6}")
    print(f"  Actual Epilepsy {cm[1,0]:>6}    {cm[1,1]:>6}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Control', 'Epilepsy']))
    
    # Save predictions
    np.savez(OUTPUT_DIR / 'test_predictions.npz',
             predictions=test_preds,
             labels=test_labels,
             probabilities=test_probs)
    
    # =============================================================================
    # VISUALIZATION
    # =============================================================================
    
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    # 1. Training history
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history['train_loss'], 'b-', linewidth=2, label='Train', alpha=0.8)
    axes[0].plot(history['val_loss'], 'r-', linewidth=2, label='Val', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], 'b-', linewidth=2, label='Train', alpha=0.8)
    axes[1].plot(history['val_acc'], 'g-', linewidth=2, label='Val', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['val_acc'], 'g-', linewidth=2, label='Accuracy')
    axes[2].plot(history['val_f1'], 'orange', linewidth=2, label='F1')
    axes[2].plot(history['val_auc'], 'm-', linewidth=2, label='AUC')
    axes[2].axhline(y=best_val_auc, color='r', linestyle='--', label=f'Best: {best_val_auc:.3f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Validation Metrics')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150)
    plt.close()
    
    # 2. Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'Epilepsy'],
                yticklabels=['Control', 'Epilepsy'])
    plt.title('Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    print("  ✓ Saved: training_history.png")
    print("  ✓ Saved: confusion_matrix.png")
    
    # =============================================================================
    # SAVE METRICS
    # =============================================================================
    
    final_metrics = {
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "best_epoch": int(checkpoint['epoch']),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_val_gap": float(checkpoint['train_val_gap'])
    }
    
    with open(OUTPUT_DIR / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Test F1:       {test_f1:.4f}")
    print(f"  Test AUC:      {test_auc:.4f}")
    print(f"\n📁 Saved to: {OUTPUT_DIR}")
    print("\n🎓 This is your MAIN thesis result: Control vs Epilepsy detection!")
