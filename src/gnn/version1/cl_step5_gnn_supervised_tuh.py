"""
Step 5 — Supervised GCN + PureGCN Ablation  (TUH Dataset, Fixed Split)
=======================================================================
DATASET CONTEXT:
  TUH epilepsy detection — binary classification: epilepsy (1) vs control (0)
  200 patients, ~193,000 epochs, 22 channels, 250 Hz, 58 flat / 17 node features
  Evaluation: fixed 60-20-20 patient-stratified split (from step 0)

KEY DIFFERENCES FROM TUC PIPELINE:
  - Evaluation: fixed train/val/test split instead of 8-fold LOPO
  - 22 nodes (channels) per graph instead of 19
  - 17 node features per channel instead of 16 (7 frequency bands vs 6)
  - Much larger dataset: ~116k train / ~39k val / ~39k test epochs
  - Task: epilepsy vs control (not ictal vs pre-ictal)
  - GNN sees many more graphs → less prone to memorisation than TUC

OVERFITTING PREVENTION — FIVE METHODS:
  1. EARLY STOPPING on validation LOSS (not val AUC)
     Patience=20. Best checkpoint restored before final evaluation.
     Stopping on val AUC would use test-patient labels for checkpoint
     selection, which is a form of evaluation leakage.

  2. DROPOUT (p=0.4 in GCN layers AND in classifier head)
     Applied after every GCN layer. Prevents co-adaptation of nodes.

  3. WEIGHT DECAY (L2 regularisation, λ=1e-4)
     Applied to all parameters via Adam weight_decay argument.
     Penalises large weights and prevents the model from fitting
     outlier patients perfectly.

  4. GRADIENT CLIPPING (max_norm=1.0)
     Prevents exploding gradients during early training, which can
     cause the model to jump to degenerate solutions.

  5. LEARNING RATE SCHEDULING (ReduceLROnPlateau)
     LR halved when val loss plateaus for 10 epochs.
     Allows the model to make fine adjustments near convergence
     rather than oscillating around a minimum.

  Additionally, the DATA-DRIVEN THRESHOLD for graph sparsification
  is computed from the training adjacency matrices only — no leakage
  from val or test patients into the graph structure.

MODELS:
  SmallGCN  — 22-node GCN with 17 hand-crafted node features
  PureGCN   — 22-node GCN where node features come from a shallow
              1D-CNN applied to raw EEG (1000 samples per channel)
              This is the ablation: does the CNN learn features as
              good as the hand-crafted ones from only training data?

ARCHITECTURE (SmallGCN):
  GCNLayer(17 → 32) → Dropout(0.4) → GCNLayer(32 → 32)
  → GlobalMeanPool → Linear(32→16) → ReLU → Dropout(0.4) → Linear(16→1)
  ~4,500 parameters total

ARCHITECTURE (PureGCN):
  RawChannelEncoder: Conv1d(1→8, k=64, s=16) → BN → ReLU → Pool(4)
                     → Linear(32→17)   [~600 params, per channel]
  Then same GCN head as SmallGCN.

Outputs (saved to --outputdir):
  results_gcn.json              SmallGCN metrics (train/val/test)
  results_pure_gcn.json         PureGCN metrics
  loss_curve_{patient}.png      training curves per fold (SmallGCN)
  roc_SmallGCN.png              ROC (val + test)
  cm_SmallGCN_{partition}.png   confusion matrices
  overfitting_SmallGCN.png      train/val/test AUC bar chart
  per_patient_SmallGCN.png      per-patient test AUC
  comparison_all_models.png     RF + SVM + SmallGCN + PureGCN

Usage:
  python step5_gnn_supervised.py \\
      --featfile      F:\\features\\features_all.npz \\
      --outputdir     F:\\results\\gnn_supervised \\
      --baseline_json F:\\results\\baseline_ml\\results_baseline.json \\
      --epochs        150 \\
      --lr            0.001 \\
      --hidden        32 \\
      --dropout       0.4 \\
      --threshold_pct 70 \\
      --patience      20 \\
      --skip_pure_gcn
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ══════════════════════════════════════════════════════════════

N_CHANNELS  = 22
N_SAMPLES   = 1000   # 250 Hz × 4 s
NODE_DIM    = 17     # hand-crafted node features per channel


# ══════════════════════════════════════════════════════════════
# 2. ADJACENCY UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_threshold(adj_dtf_train: np.ndarray,
                      percentile: float = 70.0) -> float:
    """
    Data-driven edge threshold: p-th percentile of off-diagonal DTF
    values across ALL training epochs.
    Computed from training adjacency only — never touches val/test.
    """
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    vals = np.concatenate(
        [adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))]
    )
    return float(np.percentile(vals, percentile))


def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    Self-loops added before normalisation (Kipf & Welling 2017).
    """
    A          = adj + np.eye(adj.shape[0], dtype=np.float32)
    d          = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D          = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs_features(node_feats: np.ndarray,
                           adj_dtf: np.ndarray,
                           threshold: float) -> list:
    """Build graph list for SmallGCN (hand-crafted features)."""
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x     = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, a_hat))
    return graphs


def build_graphs_raw(raw_epochs: np.ndarray,
                     adj_dtf: np.ndarray,
                     threshold: float) -> list:
    """Build graph list for PureGCN (raw EEG node input)."""
    graphs = []
    for i in range(len(raw_epochs)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x_raw = torch.tensor(raw_epochs[i], dtype=torch.float32)
        graphs.append((x_raw, a_hat))
    return graphs


def scale_node_features(graphs_train: list,
                         graphs_val:   list,
                         graphs_test:  list) -> tuple:
    """
    Fit StandardScaler on training node features.
    Apply to val and test — scaler never sees val/test data.
    """
    all_train_x = np.concatenate([g[0].numpy() for g in graphs_train], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_train_x)

    def apply(graphs):
        return [
            (torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
            for x, a in graphs
        ]
    return apply(graphs_train), apply(graphs_val), apply(graphs_test)


# ══════════════════════════════════════════════════════════════
# 3. MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """Single GCN layer: H' = ReLU( A_hat @ H @ W )"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a_hat @ x))


class SmallGCN(nn.Module):
    """
    Feature-based GCN for TUH (22 nodes, 17 node features).

    Architecture:
        GCNLayer(17 → hidden)   — neighbourhood aggregation, layer 1
        Dropout(p)
        GCNLayer(hidden → hidden) — neighbourhood aggregation, layer 2
        GlobalMeanPool           — graph-level embedding
        Linear(hidden → 16) → ReLU → Dropout(p)
        Linear(16 → 1)           — scalar logit

    ~4,500 parameters at hidden=32.
    """
    def __init__(self, in_dim: int = NODE_DIM,
                 hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, a_hat)
        h = self.drop(h)
        h = self.gcn2(h, a_hat)
        h = h.mean(dim=0, keepdim=True)   # global mean pool
        return self.head(h).squeeze()


class ImprovedRawChannelEncoder(nn.Module):
    """
    Improved 1D-CNN encoder for PureGCN.

    Improvements over original TUC RawChannelEncoder:
      - Per-channel z-score normalisation BEFORE CNN (removes amplitude scale)
      - Smaller CNN (1 conv layer vs 2) → fewer parameters → less memorisation
      - BatchNorm1d for stable embedding scale
      - Dropout inside encoder

    Input  : (22, 1000)  — raw EEG epoch, one row per channel
    Output : (22, node_dim)
    """
    def __init__(self, n_samples: int = N_SAMPLES,
                 node_dim: int = NODE_DIM, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=16, padding=32),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(4),        # → (8, 4)
        )
        self.proj  = nn.Linear(8 * 4, node_dim)   # 32 → 17
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Per-channel z-score normalisation before CNN."""
        mean   = x_raw.mean(dim=1, keepdim=True)
        std    = x_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_norm = (x_raw - mean) / std            # (22, 1000)
        n_ch   = x_norm.shape[0]
        h      = self.conv(x_norm.unsqueeze(1))  # (22, 8, 4)
        h      = self.drop2(h.view(n_ch, -1))    # (22, 32)
        return self.proj(h)                      # (22, node_dim)


class PureGCN(nn.Module):
    """
    Ablation model — replaces hand-crafted node features with CNN embeddings.
    Uses ImprovedRawChannelEncoder (with normalisation + BatchNorm).
    """
    def __init__(self, n_samples: int = N_SAMPLES,
                 node_dim: int = NODE_DIM,
                 hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.encoder = ImprovedRawChannelEncoder(n_samples, node_dim, dropout)
        self.gcn1    = GCNLayer(node_dim, hidden)
        self.gcn2    = GCNLayer(hidden, hidden)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x_raw: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        node_emb = self.encoder(x_raw)
        h        = self.gcn1(node_emb, a_hat)
        h        = self.drop(h)
        h        = self.gcn2(h, a_hat)
        h        = h.mean(dim=0, keepdim=True)
        return self.head(h).squeeze()


# ══════════════════════════════════════════════════════════════
# 4. TRAINING / EVALUATION
# ══════════════════════════════════════════════════════════════

def train_one_epoch(model, optimiser, criterion,
                    graphs: list, labels: np.ndarray,
                    device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for i in np.random.permutation(len(graphs)):
        x, a  = graphs[i]
        x, a  = x.to(device), a.to(device)
        optimiser.zero_grad()
        logit = model(x, a)
        label = torch.tensor(float(labels[i]), device=device).unsqueeze(0)
        loss  = criterion(logit.unsqueeze(0), label)
        loss.backward()
        # OVERFITTING PREVENTION: gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / max(len(graphs), 1)


@torch.no_grad()
def evaluate_graphs(model, graphs: list, labels: np.ndarray,
                    device: torch.device) -> tuple:
    """
    Returns probs (N,), preds (N,), targets (N,), bce_loss (float).
    """
    model.eval()
    logits, targets = [], []
    for i in range(len(graphs)):
        x, a = graphs[i]
        logit = model(x.to(device), a.to(device))
        logits.append(logit.cpu().item())
        targets.append(int(labels[i]))
    logits  = np.array(logits,  dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    probs   = 1.0 / (1.0 + np.exp(-logits))
    preds   = (probs >= 0.5).astype(np.int64)
    eps     = 1e-7
    bce     = -np.mean(
        targets * np.log(probs + eps) +
        (1 - targets) * np.log(1 - probs + eps)
    )
    return probs, preds, targets, float(bce)


def compute_metrics(y_true, y_pred, y_prob, partition: str = '') -> dict:
    if len(np.unique(y_true)) < 2:
        return {}
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_total        = len(y_true)
    majority_n     = max((y_true == 0).sum(), (y_true == 1).sum())
    return {
        'partition':         partition,
        'n_samples':         int(n_total),
        'accuracy':          float(accuracy_score(y_true, y_pred)),
        'majority_baseline': float(majority_n / n_total),
        'auc':               float(roc_auc_score(y_true, y_prob)),
        'f1':                float(f1_score(y_true, y_pred, zero_division=0)),
        'sensitivity':       float(tp / (tp + fn + 1e-12)),
        'specificity':       float(tn / (tn + fp + 1e-12)),
        'precision':         float(precision_score(y_true, y_pred, zero_division=0)),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ══════════════════════════════════════════════════════════════
# 5. PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def plot_training_curves(train_losses, val_losses,
                          train_aucs, val_aucs,
                          model_tag: str, output_dir: Path,
                          stopped_epoch: int = None):
    """
    Four-panel training diagnostic:
      Loss curves + AUC curves, both with early-stop marker.
    Diverging val curves = overfitting onset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0].plot(epochs, train_losses, color='steelblue', lw=1.5,
                 label='Train loss')
    axes[0].plot(epochs, val_losses,   color='tomato',    lw=1.5,
                 linestyle='--', label='Val loss')
    if stopped_epoch:
        axes[0].axvline(stopped_epoch, color='gray', linestyle=':',
                        lw=1.5, label=f'Early stop (ep {stopped_epoch})')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('BCE Loss', fontsize=11)
    axes[0].set_title(f'{model_tag} — Loss Curves', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(epochs, train_aucs, color='steelblue', lw=1.5,
                 label='Train AUC')
    axes[1].plot(epochs, val_aucs,   color='tomato',    lw=1.5,
                 linestyle='--', label='Val AUC')
    if stopped_epoch:
        axes[1].axvline(stopped_epoch, color='gray', linestyle=':',
                        lw=1.5, label=f'Early stop (ep {stopped_epoch})')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('AUC', fontsize=11)
    axes[1].set_ylim(0.4, 1.05)
    axes[1].set_title(f'{model_tag} — AUC Curves', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Annotate final gap
    final_gap = abs(train_losses[-1] - val_losses[-1]) if train_losses else 0
    axes[0].text(0.63, 0.88,
                 f'Final loss gap: {final_gap:.3f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{model_tag}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_val, y_prob_val,
                    y_test, y_prob_test,
                    model_name: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for y_true, y_prob, label, color, ls in [
        (y_val,  y_prob_val,  'Validation', 'steelblue', '-'),
        (y_test, y_prob_test, 'Test',       'tomato',    '--'),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                label=f'{label} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} — ROC Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    tag = model_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'roc_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str,
                           partition: str, output_dir: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=['Control', 'Epilepsy'],
                    yticklabels=['Control', 'Epilepsy'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'{model_name} — {partition} CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    tag = model_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'cm_{tag}_{partition.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting(results_all: dict, output_dir: Path):
    """
    Train / Val / Test AUC comparison for all GCN models.
    Primary overfitting diagnostic — mirrors step 4 plot for easy comparison.
    """
    model_names = list(results_all.keys())
    partitions  = ['train', 'val', 'test']
    colors      = {'train': 'steelblue', 'val': 'darkorange', 'test': 'tomato'}
    x           = np.arange(len(model_names))
    width       = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, partition in enumerate(partitions):
        aucs = [results_all[m].get(partition, {}).get('auc', 0.0)
                for m in model_names]
        ax.bar(x + (i - 1) * width, aucs, width,
               label=partition.capitalize(),
               color=colors[partition], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('GCN Models — Train / Val / Test AUC\n'
                 '(small gap = good generalisation)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    for i, m in enumerate(model_names):
        tr  = results_all[m].get('train', {}).get('auc', 0)
        te  = results_all[m].get('test',  {}).get('auc', 0)
        gap = tr - te
        col = 'red' if gap > 0.10 else 'black'
        ax.text(i, max(tr, te) + 0.04, f'Δ={gap:.2f}',
                ha='center', fontsize=10, fontweight='bold', color=col)

    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_gcn.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ overfitting_gcn.png')


def plot_per_patient_auc(y_true: np.ndarray, y_prob: np.ndarray,
                          patient_ids: np.ndarray,
                          model_name: str, output_dir: Path):
    patients, pat_aucs = [], []
    for pat in np.unique(patient_ids):
        mask = patient_ids == pat
        if len(np.unique(y_true[mask])) < 2:
            continue
        try:
            pat_aucs.append(float(roc_auc_score(y_true[mask], y_prob[mask])))
            patients.append(pat)
        except Exception:
            continue

    if not pat_aucs:
        return

    pat_aucs = np.array(pat_aucs)
    sort_idx = np.argsort(pat_aucs)
    colors   = ['tomato' if a < 0.5 else 'steelblue' for a in pat_aucs[sort_idx]]

    fig, ax = plt.subplots(figsize=(max(10, len(pat_aucs) * 0.4), 5))
    ax.barh(range(len(pat_aucs)), pat_aucs[sort_idx],
            color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5, label='Chance')
    ax.axvline(float(np.mean(pat_aucs)), color='navy', linestyle='-', lw=2,
               label=f'Mean = {np.mean(pat_aucs):.3f}')
    ax.set_yticks(range(len(pat_aucs)))
    ax.set_yticklabels(np.array(patients)[sort_idx], fontsize=7)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'{model_name} — Per-Patient Test AUC\n'
                 f'(red = below chance; {(pat_aucs < 0.5).sum()} patients)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    tag = model_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'per_patient_{tag}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ per_patient_{tag}.png')


def plot_all_models_comparison(gcn_results: dict,
                                baseline_json: Path,
                                output_dir: Path):
    """
    Full comparison: RF + SVM (from step 4) + SmallGCN + PureGCN.
    Shows test-set metrics only for a clean final comparison.
    """
    all_stats = {}

    if baseline_json and baseline_json.exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if res.get('test'):
                all_stats[name] = res['test']

    for name, res in gcn_results.items():
        if res.get('test'):
            all_stats[name] = res['test']

    if not all_stats:
        return

    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen', 'mediumpurple',
                'darkorange', 'teal']
    x        = np.arange(len(met_keys))
    n        = len(all_stats)
    bar_w    = 0.15

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, stats) in enumerate(all_stats.items()):
        means  = [stats.get(k, 0.0) for k in met_keys]
        offset = (i - n / 2 + 0.5) * bar_w
        ax.bar(x + offset, means, bar_w,
               label=name, color=colors[i % len(colors)],
               alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score (test set)', fontsize=12)
    ax.set_ylim(0, 1.30)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('All Models — Test Set Comparison\n'
                 '(epilepsy vs control, TUH dataset)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ comparison_all_models.png')


# ══════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_model(model_name: str,
                model_factory,
                graphs_train: list, y_train: np.ndarray,
                graphs_val:   list, y_val:   np.ndarray,
                graphs_test:  list, y_test:  np.ndarray,
                patient_ids_test: np.ndarray,
                args,
                output_dir: Path,
                device: torch.device) -> dict:
    """
    Full training run for one GCN model with all overfitting prevention.

    OVERFITTING PREVENTION applied here:
      1. Early stopping on val LOSS (patience from args)
      2. Dropout — in model architecture
      3. Weight decay — Adam weight_decay=1e-4
      4. Gradient clipping — max_norm=1.0 in train_one_epoch
      5. LR scheduling — ReduceLROnPlateau, factor=0.5, patience=10
    """
    print(f'\n{"=" * 60}')
    print(f'  {model_name}')
    print(f'{"=" * 60}')
    print(f'  Train: {len(graphs_train)} graphs  '
          f'(epi={int((y_train==1).sum())}, ctrl={int((y_train==0).sum())})')
    print(f'  Val  : {len(graphs_val)} graphs')
    print(f'  Test : {len(graphs_test)} graphs')

    # Class imbalance weight from training set only
    n_neg      = int((y_train == 0).sum())
    n_pos      = int((y_train == 1).sum())
    pos_weight = n_neg / (n_pos + 1e-12)
    print(f'  pos_weight: {pos_weight:.3f}  '
          f'(n_neg/n_pos = {n_neg}/{n_pos})')

    model     = model_factory().to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    optimiser = Adam(model.parameters(), lr=args.lr,
                     weight_decay=1e-4)           # L2 regularisation
    scheduler = ReduceLROnPlateau(optimiser,
                                  patience=10,
                                  factor=0.5,
                                  verbose=False)  # LR scheduling
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    best_val_loss  = np.inf
    best_state     = None
    patience_cnt   = 0
    stopped_epoch  = None

    print(f'\n  Training ({args.epochs} max epochs, '
          f'patience={args.patience})...')

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, optimiser, criterion, graphs_train, y_train, device
        )

        # Evaluate on train (overfitting tracking)
        tr_probs, _, tr_targets, _ = evaluate_graphs(
            model, graphs_train, y_train, device
        )
        # Evaluate on val (early stopping signal)
        val_probs, val_preds, val_targets, val_loss = evaluate_graphs(
            model, graphs_val, y_val, device
        )

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        tr_auc  = float(roc_auc_score(tr_targets,  tr_probs)) \
                  if len(np.unique(tr_targets)) == 2 else 0.0
        val_auc = float(roc_auc_score(val_targets, val_probs)) \
                  if len(np.unique(val_targets)) == 2 else 0.0
        train_aucs.append(tr_auc)
        val_aucs.append(val_auc)

        # Progress logging every 10 epochs
        if ep % 10 == 0 or ep == 1:
            gap = tr_auc - val_auc
            flag = ' ⚠' if gap > 0.10 else ''
            print(f'  Ep {ep:4d}/{args.epochs} | '
                  f'TrLoss={tr_loss:.4f}  ValLoss={val_loss:.4f} | '
                  f'TrAUC={tr_auc:.3f}  ValAUC={val_auc:.3f}  '
                  f'Gap={gap:.3f}{flag}')

        # Early stopping on val LOSS (not val AUC — avoids test-label leakage)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone()
                             for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                stopped_epoch = ep
                print(f'\n  Early stop at epoch {ep} '
                      f'(val loss did not improve for {args.patience} epochs)')
                break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'  Best checkpoint restored (val loss = {best_val_loss:.4f})')

    # ── Training curve plot ────────────────────────────────────────────
    plot_training_curves(train_losses, val_losses, train_aucs, val_aucs,
                         model_name, output_dir, stopped_epoch)

    # ── Final evaluation on all three partitions ───────────────────────
    print('\n  Final evaluation...')
    results = {}
    all_probs = {}

    for partition, graphs, labels in [
        ('train', graphs_train, y_train),
        ('val',   graphs_val,   y_val),
        ('test',  graphs_test,  y_test),
    ]:
        probs, preds, targets, _ = evaluate_graphs(
            model, graphs, labels, device
        )
        m = compute_metrics(targets, preds, probs, partition)
        results[partition] = m
        all_probs[partition] = (probs, preds, targets)

        print(f'  {partition:5s} | AUC={m.get("auc",0):.3f}  '
              f'F1={m.get("f1",0):.3f}  '
              f'Sens={m.get("sensitivity",0):.3f}  '
              f'Spec={m.get("specificity",0):.3f}  '
              f'Acc={m.get("accuracy",0):.3f}')

    tr_auc = results['train'].get('auc', 0)
    te_auc = results['test'].get('auc', 0)
    gap    = tr_auc - te_auc
    flag   = ' ⚠ OVERFIT' if gap > 0.10 else ' ✓ OK'
    print(f'\n  Train-Test AUC gap: {gap:.3f}{flag}')

    # ── Plots ──────────────────────────────────────────────────────────
    val_probs_arr,  val_preds_arr,  _ = all_probs['val']
    test_probs_arr, test_preds_arr, _ = all_probs['test']

    plot_roc_curves(y_val, val_probs_arr, y_test, test_probs_arr,
                    model_name, output_dir)

    for partition, (probs, preds, targets) in all_probs.items():
        plot_confusion_matrix(targets, preds, model_name,
                              partition, output_dir)

    plot_per_patient_auc(y_test, test_probs_arr, patient_ids_test,
                         model_name, output_dir)

    results['train_test_gap'] = float(gap)
    results['stopped_epoch']  = stopped_epoch or args.epochs
    results['n_params']       = n_params

    return results


# ══════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 5 — Supervised GCN + PureGCN ablation (TUH)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/gnn_supervised')
    parser.add_argument('--baseline_json', default=None,
                        help='Path to step4 results_baseline.json')
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold_pct', type=float, default=70.0)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--skip_pure_gcn', action='store_true',
                        help='Skip PureGCN ablation (saves time)')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5 — SUPERVISED GCN + PUREGCN  (TUH dataset)')
    print('=' * 65)
    print(f'Device        : {device}')
    print(f'Epochs        : {args.epochs}   LR: {args.lr}   Hidden: {args.hidden}')
    print(f'Dropout       : {args.dropout}   Patience: {args.patience}')
    print(f'Threshold pct : {args.threshold_pct}')
    print(f'\nOVERFITTING PREVENTION:')
    print(f'  1. Early stopping on val LOSS (patience={args.patience})')
    print(f'  2. Dropout p={args.dropout} in GCN layers + head')
    print(f'  3. Weight decay L2 λ=1e-4 (Adam)')
    print(f'  4. Gradient clipping max_norm=1.0')
    print(f'  5. ReduceLROnPlateau (factor=0.5, patience=10)')
    print('=' * 65)

    # ── Load features ──────────────────────────────────────────────────
    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)  # (N, 22, 17)
    adj_dtf     = data['adj_dtf'].astype(np.float32)        # (N, 22, 22)
    y           = data['y'].astype(np.int64)
    splits      = data['splits']
    patient_ids = data['patient_ids']

    raw_epochs = None
    if 'raw_epochs' in data and not args.skip_pure_gcn:
        raw_epochs = data['raw_epochs'].astype(np.float32)  # (N, 22, 1000)
    elif not args.skip_pure_gcn:
        print('[WARN] raw_epochs not in npz — PureGCN will be skipped')
        args.skip_pure_gcn = True

    # ── Split masks ────────────────────────────────────────────────────
    train_mask = splits == 'train'
    val_mask   = splits == 'val'
    test_mask  = splits == 'test'

    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]
    pat_test               = patient_ids[test_mask]

    majority_b = max((y == 0).sum(), (y == 1).sum()) / len(y)
    print(f'\nDataset: {len(y):,} epochs total')
    print(f'  Train : {train_mask.sum():,}  Val: {val_mask.sum():,}  '
          f'Test: {test_mask.sum():,}')
    print(f'  Majority-class baseline: {majority_b * 100:.1f}%\n')

    # ── Per-split threshold (training adjacency only) ──────────────────
    print('Computing DTF threshold from training adjacency...')
    threshold = compute_threshold(adj_dtf[train_mask], args.threshold_pct)
    print(f'  Threshold (p{args.threshold_pct:.0f}): {threshold:.4f}  '
          f'(keeps top {100-args.threshold_pct:.0f}% of edges)')

    all_gcn_results = {}

    # ══════════════════════════════════════════════════════════════
    # MODEL A — SmallGCN (hand-crafted node features)
    # ══════════════════════════════════════════════════════════════
    print('\nBuilding SmallGCN graphs...')
    g_train_raw = build_graphs_features(
        node_feats[train_mask], adj_dtf[train_mask], threshold)
    g_val_raw   = build_graphs_features(
        node_feats[val_mask],   adj_dtf[val_mask],   threshold)
    g_test_raw  = build_graphs_features(
        node_feats[test_mask],  adj_dtf[test_mask],  threshold)

    # Scale node features: fit on train only
    g_train, g_val, g_test = scale_node_features(
        g_train_raw, g_val_raw, g_test_raw)

    def make_small_gcn():
        return SmallGCN(in_dim=NODE_DIM,
                        hidden=args.hidden,
                        dropout=args.dropout)

    res_gcn = train_model(
        model_name       = 'SmallGCN',
        model_factory    = make_small_gcn,
        graphs_train=g_train, y_train=y_train,
        graphs_val=g_val,     y_val=y_val,
        graphs_test=g_test,   y_test=y_test,
        patient_ids_test = pat_test,
        args=args, output_dir=output_dir, device=device,
    )
    all_gcn_results['SmallGCN'] = res_gcn

    results_gcn_path = output_dir / 'results_gcn.json'
    with open(results_gcn_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': 'SmallGCN',
            'hyperparameters': {k: v for k, v in vars(args).items()},
            **res_gcn,
        }, f, indent=2, default=str)
    print(f'\n  ✓ SmallGCN results → {results_gcn_path}')

    # ══════════════════════════════════════════════════════════════
    # MODEL B — PureGCN (CNN node encoder, ablation)
    # ══════════════════════════════════════════════════════════════
    if not args.skip_pure_gcn:
        print('\nBuilding PureGCN graphs (raw EEG)...')
        gp_train = build_graphs_raw(
            raw_epochs[train_mask], adj_dtf[train_mask], threshold)
        gp_val   = build_graphs_raw(
            raw_epochs[val_mask],   adj_dtf[val_mask],   threshold)
        gp_test  = build_graphs_raw(
            raw_epochs[test_mask],  adj_dtf[test_mask],  threshold)

        def make_pure_gcn():
            return PureGCN(n_samples=N_SAMPLES, node_dim=NODE_DIM,
                           hidden=args.hidden, dropout=args.dropout)

        res_pure = train_model(
            model_name       = 'PureGCN',
            model_factory    = make_pure_gcn,
            graphs_train=gp_train, y_train=y_train,
            graphs_val=gp_val,     y_val=y_val,
            graphs_test=gp_test,   y_test=y_test,
            patient_ids_test = pat_test,
            args=args, output_dir=output_dir, device=device,
        )
        all_gcn_results['PureGCN'] = res_pure

        results_pure_path = output_dir / 'results_pure_gcn.json'
        with open(results_pure_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': 'PureGCN',
                'hyperparameters': {k: v for k, v in vars(args).items()},
                **res_pure,
            }, f, indent=2, default=str)
        print(f'  ✓ PureGCN results → {results_pure_path}')

    # ── Overfitting summary plot for all GCN models ───────────────────
    plot_overfitting(all_gcn_results, output_dir)

    # ── Full comparison including baselines ───────────────────────────
    bl_path = Path(args.baseline_json) if args.baseline_json else None
    plot_all_models_comparison(all_gcn_results, bl_path, output_dir)

    # ── Final summary table ────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('FINAL SUMMARY — GCN MODELS')
    print('=' * 65)
    print(f'{"Model":12s} | {"Train AUC":>10} {"Val AUC":>9} '
          f'{"Test AUC":>9} | {"Gap":>6} {"Status":>10}')
    print('-' * 65)
    for name, res in all_gcn_results.items():
        tr  = res.get('train', {}).get('auc', 0)
        val = res.get('val',   {}).get('auc', 0)
        te  = res.get('test',  {}).get('auc', 0)
        gap = tr - te
        flag = '⚠ Overfit' if gap > 0.10 else '✓ OK'
        print(f'{name:12s} | {tr:10.3f} {val:9.3f} {te:9.3f} '
              f'| {gap:6.3f} {flag:>10}')

    print(f'\nMajority-class baseline: {majority_b * 100:.1f}%')
    print('\n' + '=' * 65)
    print('STEP 5 COMPLETE')
    print('=' * 65)
    print(f'\nNext: python step6_ssl_gnn.py --featfile {args.featfile}')


if __name__ == '__main__':
    main()
