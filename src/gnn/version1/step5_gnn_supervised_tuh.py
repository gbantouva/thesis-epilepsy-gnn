"""
Step 5 — Supervised GCN + Adjacency-Only GCN + PureGCN  (TUH Dataset)
======================================================================
DATASET CONTEXT:
  TUH epilepsy detection — binary: epilepsy (1) vs control (0)
  200 patients, ~193k epochs, 22 channels, 250 Hz
  58 flat / 17 node features
  Evaluation: fixed 60-20-20 patient-stratified split (from step 0)

KEY DIFFERENCES FROM TUC PIPELINE:
  - Evaluation: fixed train/val/test split instead of 8-fold nested LOPO.
    With 200 patients, LOPO would require 200 full training runs including
    200 SSL pre-trainings — computationally unreasonable. A fixed
    patient-stratified split with strict patient separation is the standard
    evaluation approach for TUH-scale datasets.
  - 22 nodes (channels) per graph instead of 19
  - 17 node features per channel instead of 16 (7 frequency bands vs 6)
  - Much larger dataset: ~116k train / ~39k val / ~39k test epochs

MODELS — THREE VARIANTS (matching TUC methodology):
  SmallGCN     — GCN with 17 hand-crafted node features (spectral, Hjorth,
                 time-domain, connectivity degrees). Primary model.
  AdjOnlyGCN   — GCN where each node's feature = its raw DTF adjacency row
                 (22 values, outgoing connectivity profile only). No spectral
                 or Hjorth features. Direct ablation matching TUC V8 design:
                 tests whether graph topology alone suffices.
  PureGCN      — GCN where node features come from a shallow 1D-CNN applied
                 to raw EEG (1000 samples per channel). Learns its own node
                 representation. Unique to TUH — on TUC this collapsed due to
                 insufficient training data (~900 graphs). On TUH (~116k graphs)
                 it may be viable. Kept for completeness.

WHY AdjOnlyGCN IS THE PRIMARY ABLATION:
  The AdjOnlyGCN is the clean ablation that directly answers: do the
  hand-crafted spectral and Hjorth node features contribute independently
  beyond what is already encoded in the graph connectivity structure?
  It uses identical architecture, training, and evaluation, differing only
  in the node feature input. This is the same design used in the FORTH
  (TUC) pipeline V8, ensuring methodological consistency across datasets.
  PureGCN is a secondary ablation testing raw-signal learning at scale.

OVERFITTING PREVENTION — FIVE METHODS:
  1. EARLY STOPPING on validation LOSS (not val AUC, patience configurable)
  2. DROPOUT p=0.4 in GCN layers and classifier head
  3. WEIGHT DECAY L2 λ=1e-4 via Adam weight_decay
  4. GRADIENT CLIPPING max_norm=1.0 every update step
  5. LEARNING RATE SCHEDULING: ReduceLROnPlateau, factor=0.5, patience=10

DATA LEAKAGE PREVENTION:
  - DTF threshold computed from training adjacency only (val/test never touched)
  - StandardScaler fitted on training node features only
  - Class imbalance weight (pos_weight) computed from training labels only
  - Val set used for early stopping only — not for model selection or tuning

Outputs (saved to --outputdir):
  results_gcn.json                SmallGCN metrics (train/val/test)
  results_gcn_adj_only.json       AdjOnlyGCN metrics
  results_pure_gcn.json           PureGCN metrics (if not --skip_pure_gcn)
  training_curves_{model}.png     loss + AUC curves per model
  roc_{model}.png                 ROC curves (val + test)
  cm_{model}_{partition}.png      confusion matrices
  overfitting_{model}.png         train/val/test AUC bar chart
  per_patient_{model}.png         per-patient test AUC
  comparison_gcn_variants.png     SmallGCN vs AdjOnlyGCN
  comparison_all_models.png       all models including baselines

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
    values across ALL training epochs. Computed from training adjacency
    only — never touches val or test patients.

    With p=70, the top 30% of connections are retained, giving
    approximately 0.30 × 22 × 21 ≈ 138 directed edges per graph.
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
    """
    Build graph list for SmallGCN (hand-crafted 17-dim node features).
    threshold: data-driven value computed from training adjacency only.
    """
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x     = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, a_hat))
    return graphs


def build_graphs_adj_only(adj_dtf: np.ndarray,
                           threshold: float) -> list:
    """
    Build graph list for AdjOnlyGCN.

    Node features = the raw (unthresholded) DTF adjacency row for each
    channel, giving a 22-dimensional connectivity profile per node.
    No spectral, Hjorth, or time-domain information is included.

    The graph structure (a_hat) uses the same thresholded, normalised
    adjacency as SmallGCN, ensuring a fair comparison of node feature
    utility under identical graph topology.

    This is the primary ablation: it tests whether the hand-crafted
    spectral and Hjorth features add discriminative information beyond
    what is already encoded in the connectivity structure.
    """
    graphs = []
    for i in range(len(adj_dtf)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        # Node features: raw (unthresholded) adjacency row
        x_raw = adj_dtf[i].copy()
        np.fill_diagonal(x_raw, 0.0)
        graphs.append((torch.tensor(x_raw.astype(np.float32)), a_hat))
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
    Fit StandardScaler on training node features only.
    Apply to val and test — scaler never sees val/test patient data.
    Not used for AdjOnlyGCN (adjacency rows are not z-score scaled,
    since they are ratio quantities bounded in [0,1]).
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
    Feature-based GCN for TUH (22 nodes, 17 hand-crafted node features).
    Architecture:
        GCNLayer(in_dim → hidden)  — neighbourhood aggregation, layer 1
        Dropout(p)
        GCNLayer(hidden → hidden)  — neighbourhood aggregation, layer 2
        GlobalMeanPool             — graph-level embedding
        Linear(hidden → 16) → ReLU → Dropout(p) → Linear(16 → 1)
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
    1D-CNN encoder for PureGCN.
    Per-channel z-score normalisation + InstanceNorm (stable with n=22 samples).
    Input: (22, 1000) — Output: (22, node_dim)
    """
    def __init__(self, n_samples: int = N_SAMPLES,
                 node_dim: int = NODE_DIM, dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=16, padding=32),
            nn.InstanceNorm1d(8, affine=True),   # stable with small n_ch
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj  = nn.Linear(8 * 4, node_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        mean   = x_raw.mean(dim=1, keepdim=True)
        std    = x_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_norm = (x_raw - mean) / std
        n_ch   = x_norm.shape[0]
        h      = self.conv(x_norm.unsqueeze(1))
        h      = self.drop2(h.view(n_ch, -1))
        return self.proj(h)


class PureGCN(nn.Module):
    """
    CNN-based node feature learning + GCN.
    Ablation: replaces hand-crafted features with raw-signal CNN embeddings.
    Uses InstanceNorm (stable with single-graph processing).
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / max(len(graphs), 1)


@torch.no_grad()
def evaluate_graphs(model, graphs: list, labels: np.ndarray,
                    device: torch.device) -> tuple:
    """Returns probs, preds, targets, bce_loss."""
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

def plot_training_curves(train_losses, val_losses, train_aucs, val_aucs,
                          model_tag: str, output_dir: Path,
                          stopped_epoch: int = None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(train_losses, color='steelblue',  lw=1.5, label='Train loss')
    ax.plot(val_losses,   color='tomato',     lw=1.5, linestyle='--', label='Val loss')
    if stopped_epoch:
        ax.axvline(stopped_epoch - 1, color='green', lw=1, linestyle=':',
                   label=f'Early stop (ep {stopped_epoch})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
    ax.set_title(f'{model_tag} — Loss', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(train_aucs, color='steelblue', lw=1.5, label='Train AUC')
    ax.plot(val_aucs,   color='tomato',    lw=1.5, linestyle='--', label='Val AUC')
    if stopped_epoch:
        ax.axvline(stopped_epoch - 1, color='green', lw=1, linestyle=':')
    if train_aucs and val_aucs:
        gap = train_aucs[-1] - val_aucs[-1]
        ax.text(0.97, 0.03, f'Final gap: {gap:+.3f}',
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f'{model_tag} — AUC', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{model_tag} — Training Diagnostic', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_{model_tag}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_val, val_probs, y_test, test_probs,
                    model_tag: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for y_true, probs, label, color in [
        (y_val,  val_probs,  'Val',  'darkorange'),
        (y_test, test_probs, 'Test', 'steelblue'),
    ]:
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_tag} — ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_tag}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_tag: str,
                           partition: str, output_dir: Path):
    cmap = 'Blues' if 'adj' not in model_tag.lower() else 'Purples'
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Control', 'Epilepsy'],
                yticklabels=['Control', 'Epilepsy'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{model_tag} | {partition}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_{model_tag}_{partition}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting(all_gcn_results: dict, output_dir: Path):
    """Train / Val / Test AUC bar chart for all GCN models."""
    models    = list(all_gcn_results.keys())
    partitions = ['train', 'val', 'test']
    colors     = ['steelblue', 'darkorange', 'tomato']
    x          = np.arange(len(models))
    w          = 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
    for j, (part, col) in enumerate(zip(partitions, colors)):
        vals = [all_gcn_results[m].get(part, {}).get('auc', 0.0) for m in models]
        ax.bar(x + (j - 1) * w, vals, w, label=part.capitalize(),
               color=col, alpha=0.85, edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, fontsize=10)
    ax.set_ylabel('AUC'); ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('GCN Models — Train / Val / Test AUC\n'
                 '(large train–test gap = overfitting)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_gcn_models.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_gcn_variants_comparison(feat_stats: dict, adj_stats: dict,
                                  output_dir: Path):
    """SmallGCN vs AdjOnlyGCN side-by-side — the primary ablation plot."""
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity', 'accuracy', 'mcc']
    x, w = np.arange(len(met_keys)), 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (label, stats, col) in enumerate([
        ('SmallGCN (features)', feat_stats, 'steelblue'),
        ('AdjOnlyGCN (topology)', adj_stats, 'mediumpurple'),
    ]):
        vals = [stats.get(k, 0.0) for k in met_keys]
        ax.bar(x + (i - 0.5) * w, vals, w, label=label,
               color=col, alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=11)
    ax.set_ylabel('Score (test set)', fontsize=12); ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('SmallGCN vs AdjOnlyGCN — Test Set (TUH)\n'
                 'AdjOnly isolates contribution of hand-crafted node features',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_gcn_variants.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ comparison_gcn_variants.png')


def plot_all_models_comparison(gcn_results: dict, baseline_json_path,
                                output_dir: Path):
    all_stats = {}
    if baseline_json_path and Path(baseline_json_path).exists():
        with open(baseline_json_path) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if isinstance(res, dict) and 'test' in res:
                all_stats[name] = res['test']
    for name, res in gcn_results.items():
        if res.get('test'):
            all_stats[name] = res['test']

    if not all_stats:
        return

    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'darkorange', 'seagreen',
                'mediumpurple', 'teal', 'coral']
    x = np.arange(len(met_keys))
    n = len(all_stats)
    w = 0.14

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, stats) in enumerate(all_stats.items()):
        means  = [stats.get(k, 0.0) for k in met_keys]
        offset = (i - n / 2 + 0.5) * w
        ax.bar(x + offset, means, w, label=name,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score (test set)', fontsize=12)
    ax.set_ylim(0, 1.30)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('All Models — Test Set Comparison (TUH dataset)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png', dpi=150, bbox_inches='tight')
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
    Full training run with all overfitting prevention measures.
    Early stopping on val LOSS (not val AUC — avoids checkpoint leakage).
    """
    print(f'\n{"=" * 60}')
    print(f'  {model_name}')
    print(f'{"=" * 60}')
    print(f'  Train: {len(graphs_train):,} graphs  '
          f'(epi={int((y_train==1).sum()):,}, ctrl={int((y_train==0).sum()):,})')
    print(f'  Val  : {len(graphs_val):,} graphs')
    print(f'  Test : {len(graphs_test):,} graphs')

    n_neg      = int((y_train == 0).sum())
    n_pos      = int((y_train == 1).sum())
    pos_weight = n_neg / (n_pos + 1e-12)
    print(f'  pos_weight: {pos_weight:.3f}  (n_neg/n_pos = {n_neg}/{n_pos})')

    model    = model_factory().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    best_val_loss = np.inf
    best_state    = None
    patience_cnt  = 0
    stopped_epoch = None

    print(f'\n  Training ({args.epochs} max epochs, patience={args.patience})...')

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, optimiser, criterion, graphs_train, y_train, device
        )
        tr_probs, _, tr_targets, _ = evaluate_graphs(
            model, graphs_train, y_train, device
        )
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

        if ep % 10 == 0 or ep == 1:
            gap  = tr_auc - val_auc
            flag = ' ⚠' if gap > 0.10 else ''
            print(f'  Ep {ep:4d}/{args.epochs} | '
                  f'TrLoss={tr_loss:.4f}  ValLoss={val_loss:.4f} | '
                  f'TrAUC={tr_auc:.3f}  ValAUC={val_auc:.3f}  '
                  f'Gap={gap:.3f}{flag}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                stopped_epoch = ep
                print(f'\n  Early stop at epoch {ep} '
                      f'(val loss unchanged for {args.patience} epochs)')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'  Best checkpoint restored (val loss = {best_val_loss:.4f})')

    plot_training_curves(train_losses, val_losses, train_aucs, val_aucs,
                         model_name, output_dir, stopped_epoch)

    print('\n  Final evaluation...')
    results   = {}
    all_probs = {}

    for partition, graphs, labels in [
        ('train', graphs_train, y_train),
        ('val',   graphs_val,   y_val),
        ('test',  graphs_test,  y_test),
    ]:
        probs, preds, targets, _ = evaluate_graphs(model, graphs, labels, device)
        m = compute_metrics(targets, preds, probs, partition)
        results[partition]  = m
        all_probs[partition] = (probs, preds, targets)
        print(f'  {partition:5s} | AUC={m.get("auc",0):.3f}  '
              f'F1={m.get("f1",0):.3f}  Sens={m.get("sensitivity",0):.3f}  '
              f'Spec={m.get("specificity",0):.3f}  Acc={m.get("accuracy",0):.3f}  '
              f'MCC={m.get("mcc",0):.3f}')

    tr_auc = results['train'].get('auc', 0)
    te_auc = results['test'].get('auc', 0)
    gap    = tr_auc - te_auc
    flag   = ' ⚠ OVERFIT' if gap > 0.10 else ' ✓ OK'
    print(f'\n  Train-Test AUC gap: {gap:.3f}{flag}')

    val_probs_arr,  val_preds_arr,  _ = all_probs['val']
    test_probs_arr, test_preds_arr, _ = all_probs['test']

    plot_roc_curves(y_val, val_probs_arr, y_test, test_probs_arr,
                    model_name, output_dir)

    for partition, (probs, preds, targets) in all_probs.items():
        plot_confusion_matrix(targets, preds, model_name, partition, output_dir)

    results['train_test_gap'] = float(gap)
    results['stopped_epoch']  = stopped_epoch or args.epochs
    results['n_params']       = n_params
    return results


# ══════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 5 — SmallGCN + AdjOnlyGCN + PureGCN (TUH)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',       required=True)
    parser.add_argument('--outputdir',      default='results/gnn_supervised')
    parser.add_argument('--baseline_json',  default=None)
    parser.add_argument('--epochs',         type=int,   default=150)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--hidden',         type=int,   default=32)
    parser.add_argument('--dropout',        type=float, default=0.4)
    parser.add_argument('--threshold_pct',  type=float, default=70.0,
                        help='DTF percentile threshold from training adjacency')
    parser.add_argument('--patience',       type=int,   default=20)
    parser.add_argument('--skip_pure_gcn',  action='store_true',
                        help='Skip PureGCN (CNN ablation). AdjOnlyGCN always runs.')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5 — SmallGCN + AdjOnlyGCN + PureGCN  (TUH dataset)')
    print('=' * 65)
    print(f'Device        : {device}')
    print(f'Epochs        : {args.epochs}   LR: {args.lr}   Hidden: {args.hidden}')
    print(f'Dropout       : {args.dropout}   Patience: {args.patience}')
    print(f'Threshold pct : {args.threshold_pct}  (data-driven from training adjacency)')
    print(f'\nModels:')
    print(f'  SmallGCN    — 17 hand-crafted node features (primary model)')
    print(f'  AdjOnlyGCN  — DTF adjacency row as node feature (primary ablation)')
    print(f'  PureGCN     — CNN node encoder from raw EEG '
          f'({"SKIPPED" if args.skip_pure_gcn else "enabled"})')
    print(f'\nOVERFITTING PREVENTION:')
    print(f'  1. Early stopping on val LOSS  (patience={args.patience})')
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
    print(f'  Epilepsy: {(y==1).sum():,}  Control: {(y==0).sum():,}')
    print(f'  Majority-class baseline: {majority_b * 100:.1f}%\n')

    # ── Data-driven threshold (training adjacency only) ────────────────
    print('Computing DTF threshold from training adjacency...')
    threshold = compute_threshold(adj_dtf[train_mask], args.threshold_pct)
    print(f'  Threshold (p{args.threshold_pct:.0f}): {threshold:.4f}  '
          f'(keeps top {100 - args.threshold_pct:.0f}% of edges)')

    all_gcn_results = {}

    # ══════════════════════════════════════════════════════════════
    # MODEL A — SmallGCN (hand-crafted node features)
    # ══════════════════════════════════════════════════════════════
    print('\nBuilding SmallGCN graphs (hand-crafted features)...')
    g_tr_raw = build_graphs_features(node_feats[train_mask], adj_dtf[train_mask], threshold)
    g_va_raw = build_graphs_features(node_feats[val_mask],   adj_dtf[val_mask],   threshold)
    g_te_raw = build_graphs_features(node_feats[test_mask],  adj_dtf[test_mask],  threshold)
    g_train, g_val, g_test = scale_node_features(g_tr_raw, g_va_raw, g_te_raw)

    res_gcn = train_model(
        'SmallGCN', lambda: SmallGCN(NODE_DIM, args.hidden, args.dropout),
        g_train, y_train, g_val, y_val, g_test, y_test,
        pat_test, args, output_dir, device,
    )
    all_gcn_results['SmallGCN'] = res_gcn

    with open(output_dir / 'results_gcn.json', 'w') as f:
        json.dump({'model': 'SmallGCN',
                   'hyperparameters': {k: v for k, v in vars(args).items()
                                       if not k.startswith('_')},
                   'threshold': threshold,
                   **res_gcn}, f, indent=2, default=str)
    print(f'  ✓ SmallGCN results → results_gcn.json')

    # ══════════════════════════════════════════════════════════════
    # MODEL B — AdjOnlyGCN (adjacency row as node feature)
    # Primary ablation matching TUC V8 methodology.
    # ══════════════════════════════════════════════════════════════
    print('\nBuilding AdjOnlyGCN graphs (DTF adjacency row as node feature)...')
    print('  Node features = raw DTF adjacency row (22 values per channel)')
    print('  No spectral, Hjorth, or time-domain features.')
    print('  Same graph structure and GCN architecture as SmallGCN.')

    ga_train = build_graphs_adj_only(adj_dtf[train_mask], threshold)
    ga_val   = build_graphs_adj_only(adj_dtf[val_mask],   threshold)
    ga_test  = build_graphs_adj_only(adj_dtf[test_mask],  threshold)
    # NOTE: no StandardScaler for AdjOnly — DTF values are already in [0,1]

    res_adj = train_model(
        'AdjOnlyGCN', lambda: SmallGCN(N_CHANNELS, args.hidden, args.dropout),
        ga_train, y_train, ga_val, y_val, ga_test, y_test,
        pat_test, args, output_dir, device,
    )
    all_gcn_results['AdjOnlyGCN'] = res_adj

    with open(output_dir / 'results_gcn_adj_only.json', 'w') as f:
        json.dump({'model': 'AdjOnlyGCN',
                   'description': 'Node features = DTF adjacency row (no hand-crafted features). '
                                  'Primary ablation matching TUC V8 methodology.',
                   'hyperparameters': {k: v for k, v in vars(args).items()
                                       if not k.startswith('_')},
                   'threshold': threshold,
                   **res_adj}, f, indent=2, default=str)
    print(f'  ✓ AdjOnlyGCN results → results_gcn_adj_only.json')

    # AdjOnly vs SmallGCN comparison plot
    feat_test = res_gcn.get('test', {})
    adj_test  = res_adj.get('test', {})
    if feat_test and adj_test:
        plot_gcn_variants_comparison(feat_test, adj_test, output_dir)

    # ══════════════════════════════════════════════════════════════
    # MODEL C — PureGCN (CNN node encoder, secondary ablation)
    # ══════════════════════════════════════════════════════════════
    if not args.skip_pure_gcn:
        print('\nBuilding PureGCN graphs (raw EEG → CNN node encoder)...')
        gp_train = build_graphs_raw(raw_epochs[train_mask], adj_dtf[train_mask], threshold)
        gp_val   = build_graphs_raw(raw_epochs[val_mask],   adj_dtf[val_mask],   threshold)
        gp_test  = build_graphs_raw(raw_epochs[test_mask],  adj_dtf[test_mask],  threshold)

        res_pure = train_model(
            'PureGCN',
            lambda: PureGCN(N_SAMPLES, NODE_DIM, args.hidden, args.dropout),
            gp_train, y_train, gp_val, y_val, gp_test, y_test,
            pat_test, args, output_dir, device,
        )
        all_gcn_results['PureGCN'] = res_pure

        with open(output_dir / 'results_pure_gcn.json', 'w') as f:
            json.dump({'model': 'PureGCN',
                       'hyperparameters': {k: v for k, v in vars(args).items()
                                           if not k.startswith('_')},
                       'threshold': threshold,
                       **res_pure}, f, indent=2, default=str)
        print(f'  ✓ PureGCN results → results_pure_gcn.json')

    # ── Aggregate plots ────────────────────────────────────────────────
    plot_overfitting(all_gcn_results, output_dir)
    plot_all_models_comparison(all_gcn_results,
                                Path(args.baseline_json) if args.baseline_json else None,
                                output_dir)

    # ── Final summary table ────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('FINAL SUMMARY — GCN MODELS (TUH TEST SET)')
    print('=' * 65)
    print(f'{"Model":15s} | {"Train AUC":>10} {"Val AUC":>9} '
          f'{"Test AUC":>9} | {"Gap":>6} {"MCC":>6} {"Status":>10}')
    print('-' * 70)
    for name, res in all_gcn_results.items():
        tr  = res.get('train', {}).get('auc', 0)
        val = res.get('val',   {}).get('auc', 0)
        te  = res.get('test',  {}).get('auc', 0)
        mcc = res.get('test',  {}).get('mcc', 0)
        gap = tr - te
        flag = '⚠ Overfit' if gap > 0.10 else '✓ OK'
        print(f'{name:15s} | {tr:10.3f} {val:9.3f} {te:9.3f} '
              f'| {gap:6.3f} {mcc:6.3f} {flag:>10}')

    print(f'\nMajority-class baseline: {majority_b * 100:.1f}%')
    print(f'DTF threshold (p{args.threshold_pct:.0f}): {threshold:.4f}')
    print('\n' + '=' * 65)
    print('STEP 5 COMPLETE')
    print('=' * 65)
    print(f'\nNext: python step6_ssl_gnn.py --featfile {args.featfile}'
          f' --sup_json {output_dir / "results_gcn.json"}')


if __name__ == '__main__':
    main()
