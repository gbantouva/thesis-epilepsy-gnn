"""
Step 5b — PureGCN (Fixed, TUH Dataset)
=======================================
PURPOSE:
  Standalone ablation script that re-runs ONLY PureGCN with three critical
  fixes for the collapse observed in step 5, plus optimal threshold selection
  to fix the sensitivity bias in the original fixed version.

  PROBLEM 1 — BatchNorm1d with effective batch size = 22
    In step 5, training processes one graph at a time. The CNN encoder
    applies BatchNorm1d across the 22 channels of that single graph.
    BatchNorm with n=22 samples produces highly unstable running mean/var
    estimates. The result: all encoder outputs converge to the same
    embedding regardless of input → GCN has no discriminative signal →
    model collapses to predicting the majority class everywhere.

    FIX: Replace BatchNorm1d with InstanceNorm1d(affine=True).
    InstanceNorm normalises within each channel independently without
    needing stable cross-sample statistics. affine=True keeps learnable
    scale/shift parameters. This is the correct normalisation choice
    when processing one sample at a time.

  PROBLEM 2 — Single-graph gradient updates
    Updating weights after every single graph produces very noisy
    gradients, especially for a CNN that needs to see diverse inputs
    to learn general temporal features.

    FIX: Gradient accumulation over mini-batches of graphs.
    Gradients are accumulated for ACCUMULATION_STEPS graphs before
    calling optimiser.step(). This simulates a larger effective batch
    size without requiring batched graph processing.

  PROBLEM 3 — Sensitivity bias from fixed threshold=0.5
    The first fixed version had specificity=0.339 and sensitivity=0.933.
    The model was predicting epilepsy for nearly every epoch because the
    optimal decision boundary is not at probability=0.5 for this model.

    FIX: Youden's J threshold selection on validation set.
    Find the threshold maximising (sensitivity + specificity - 1) on the
    val set only. Apply to test. No test-set leakage — the threshold is
    derived exclusively from val patients.

  ADDITIONALLY — per-channel z-score normalisation inside the encoder
    forward pass removes patient-specific amplitude scale before the CNN,
    which is the most important single fix for cross-patient generalisation.

WHAT THIS TESTS:
  Can a shallow CNN learn node embeddings from raw EEG that are as
  discriminative as hand-crafted spectral/Hjorth features?

  On TUC (small dataset, ~900 training graphs):
    No — the CNN cannot learn cross-patient features from so few examples.
    PureGCN underfits or collapses.

  On TUH (large dataset, ~52k training graphs):
    Maybe — with enough data the CNN may learn useful temporal features.
    The result directly determines whether hand-crafted features are
    necessary or whether raw signal learning is sufficient at scale.
    This is the central thesis argument for the ablation.

EVALUATION:
  Same fixed 60-20-20 split as step 5. Same GCN head architecture.
  Threshold selected on val set, evaluated on test set.
  The only difference from SmallGCN is the node feature source.

Outputs (saved to --outputdir):
  results_pure_gcn_fixed.json      all metrics + optimal threshold
  loss_curve_PureGCN_fixed.png     training curves
  roc_puregcn_fixed.png            ROC curves (val + test) with threshold marker
  cm_puregcn_fixed_{partition}.png confusion matrices at optimal threshold
  overfitting_puregcn_fixed.png    train/val/test AUC bar chart
  per_patient_puregcn_fixed.png    per-patient test AUC
  comparison_versions.png          PureGCN original vs fixed vs SmallGCN

Usage:
  python step5b_pure_gcn_tuh.py \\
      --featfile        F:\\features\\features_all.npz \\
      --outputdir       F:\\results\\pure_gcn_fixed \\
      --original_json   F:\\results\\gnn_supervised\\results_pure_gcn.json \\
      --sup_json        F:\\results\\gnn_supervised\\results_gcn.json \\
      --epochs          150 \\
      --lr              0.001 \\
      --hidden          32 \\
      --dropout         0.4 \\
      --threshold_pct   50 \\
      --patience        20 \\
      --accum_steps     8
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

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

N_CHANNELS = 22
N_SAMPLES  = 1000
NODE_DIM   = 17


# ══════════════════════════════════════════════════════════════
# ADJACENCY UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_threshold(adj_dtf_train: np.ndarray,
                      percentile: float = 50.0) -> float:
    """
    Data-driven edge threshold from training adjacency only.

    For TUH resting-state EEG, percentile=50 (keep top 50% of edges)
    is used instead of 70% from step 5. TUH connectivity matrices are
    sparser on average — keeping only the top 30% risks near-empty graphs
    for control patients with weak resting-state connectivity.
    """
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    vals = np.concatenate(
        [adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))]
    )
    return float(np.percentile(vals, percentile))


def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """A_hat = D^{-1/2} (A + I) D^{-1/2}  (Kipf & Welling, 2017)"""
    A          = adj + np.eye(adj.shape[0], dtype=np.float32)
    d          = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D          = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs_raw(raw_epochs: np.ndarray,
                     adj_dtf: np.ndarray,
                     threshold: float) -> list:
    """Build (x_raw, a_hat) graph list for PureGCN."""
    graphs = []
    for i in range(len(raw_epochs)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x_raw = torch.tensor(raw_epochs[i], dtype=torch.float32)
        graphs.append((x_raw, a_hat))
    return graphs


# ══════════════════════════════════════════════════════════════
# THRESHOLD SELECTION
# ══════════════════════════════════════════════════════════════

def find_optimal_threshold(y_val: np.ndarray,
                            val_probs: np.ndarray) -> tuple:
    """
    Find the classification threshold that maximises Youden's J statistic
    on the validation set.

    Youden's J = sensitivity + specificity - 1
               = TPR - FPR

    This is the point on the ROC curve geometrically closest to the
    top-left corner (perfect classifier). It balances sensitivity and
    specificity equally, which is appropriate here because both false
    negatives (missed epilepsy) and false positives (misclassified
    control) have clinical consequences.

    LEAKAGE-FREE: threshold is computed on val patients only.
    It is then applied to the test set without any further adjustment.

    Parameters
    ----------
    y_val      : (N,) ground truth labels for validation set
    val_probs  : (N,) predicted probabilities for validation set

    Returns
    -------
    best_threshold : float
    best_sensitivity : float  (at this threshold, on val)
    best_specificity : float  (at this threshold, on val)
    """
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    j_scores  = tpr - fpr           # Youden's J per threshold
    best_idx  = int(np.argmax(j_scores))
    best_thr  = float(thresholds[best_idx])
    best_sens = float(tpr[best_idx])
    best_spec = float(1.0 - fpr[best_idx])

    print(f'  Optimal threshold (Youden J on val): {best_thr:.4f}')
    print(f'    Val sensitivity at threshold : {best_sens:.3f}')
    print(f'    Val specificity at threshold : {best_spec:.3f}')
    print(f'    Val Youden J                 : {best_sens + best_spec - 1:.3f}')

    return best_thr, best_sens, best_spec


# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class FixedRawChannelEncoder(nn.Module):
    """
    FIXED 1D-CNN encoder for raw EEG per channel.

    FIX 1 — InstanceNorm1d instead of BatchNorm1d
      When processing one graph at a time, BatchNorm1d sees only 22
      samples (one per channel). This is too few for stable estimates.
      InstanceNorm1d normalises within each of the 8 feature maps
      independently, requiring no cross-sample statistics.
      affine=True preserves learnable scale and shift parameters.

    FIX 2 — Per-channel z-score normalisation BEFORE CNN
      Removes patient-specific amplitude scale before the CNN sees the
      signal. Forces the CNN to learn shape-based temporal features
      (oscillation patterns, transients) rather than absolute amplitude.
      This is the most important fix for cross-patient generalisation.

    Architecture:
      Input  : (22, 1000) — one row per EEG channel
      z-score per channel → (22, 1000)
      Conv1d(1→8, k=64, stride=16, pad=32) → InstanceNorm → ReLU → Dropout
      AdaptiveAvgPool1d(4)  → (22, 8, 4)
      flatten → (22, 32)
      Linear(32 → node_dim) → (22, 17)
    """
    def __init__(self, n_samples: int = N_SAMPLES,
                 node_dim: int = NODE_DIM,
                 dropout: float = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=16, padding=32),
            nn.InstanceNorm1d(8, affine=True),   # FIX 1: not BatchNorm1d
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj  = nn.Linear(8 * 4, node_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # FIX 2: per-channel z-score normalisation
        mean   = x_raw.mean(dim=1, keepdim=True)
        std    = x_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_norm = (x_raw - mean) / std             # (22, 1000)
        n_ch   = x_norm.shape[0]
        h      = self.conv(x_norm.unsqueeze(1))   # (22, 8, 4)
        h      = self.drop2(h.view(n_ch, -1))     # (22, 32)
        return self.proj(h)                        # (22, node_dim)


class GCNLayer(nn.Module):
    """Single GCN layer: H' = ReLU( A_hat @ H @ W )"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a @ x))


class FixedPureGCN(nn.Module):
    """
    FIXED PureGCN — raw EEG → CNN encoder → GCN → binary classifier.

    GCN head is identical to SmallGCN in step 5 for a fair ablation.
    The only difference from SmallGCN is the node feature source:
      SmallGCN  : hand-crafted spectral + Hjorth + connectivity features
      FixedPureGCN : CNN-learned embeddings from raw EEG signal

    Architecture:
      FixedRawChannelEncoder  (22, 1000) → (22, NODE_DIM)
      GCNLayer(NODE_DIM → hidden) → Dropout
      GCNLayer(hidden → hidden)
      GlobalMeanPool           → (hidden,)
      Linear(hidden → 16) → ReLU → Dropout
      Linear(16 → 1)           — scalar logit
    """
    def __init__(self, n_samples: int = N_SAMPLES,
                 node_dim: int = NODE_DIM,
                 hidden: int = 32,
                 dropout: float = 0.4):
        super().__init__()
        self.encoder = FixedRawChannelEncoder(n_samples, node_dim, dropout)
        self.gcn1    = GCNLayer(node_dim, hidden)
        self.gcn2    = GCNLayer(hidden, hidden)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x_raw: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        node_emb = self.encoder(x_raw)
        h        = self.gcn1(node_emb, a)
        h        = self.drop(h)
        h        = self.gcn2(h, a)
        h        = h.mean(dim=0, keepdim=True)   # global mean pool
        return self.head(h).squeeze()


# ══════════════════════════════════════════════════════════════
# TRAINING WITH GRADIENT ACCUMULATION
# ══════════════════════════════════════════════════════════════

def train_one_epoch_accum(model: nn.Module,
                           optimiser,
                           criterion,
                           graphs: list,
                           labels: np.ndarray,
                           device: torch.device,
                           accum_steps: int = 8) -> float:
    """
    Train one epoch with gradient accumulation.

    FIX 3 — Gradient accumulation over mini-batches of graphs.

    Updating weights after every single graph produces very noisy
    gradients. The CNN encoder needs to see diverse inputs within each
    update to learn general temporal features rather than fitting
    individual graphs. Accumulating over accum_steps graphs before
    each optimiser step simulates a mini-batch of that size without
    requiring memory for a full batch.

    The loss is divided by accum_steps before backward so the
    accumulated gradient equals the mean gradient over the mini-batch.
    """
    model.train()
    total_loss = 0.0
    perm       = np.random.permutation(len(graphs))
    optimiser.zero_grad()

    for step, i in enumerate(perm):
        x, a  = graphs[i]
        x, a  = x.to(device), a.to(device)
        logit = model(x, a)
        label = torch.tensor(float(labels[i]),
                             device=device).unsqueeze(0)
        loss  = criterion(logit.unsqueeze(0), label) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()

    # Final update for remaining graphs that did not complete a full batch
    remaining = len(graphs) % accum_steps
    if remaining > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        optimiser.zero_grad()

    return total_loss / max(len(graphs), 1)


@torch.no_grad()
def evaluate_graphs(model: nn.Module,
                    graphs: list,
                    labels: np.ndarray,
                    device: torch.device) -> tuple:
    """
    Evaluate model on a graph list.
    Returns: probs (N,), targets (N,), bce_loss (float).
    Note: predictions at threshold=0.5 are NOT returned here.
    The caller applies the optimal threshold externally.
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
    eps     = 1e-7
    bce     = -np.mean(
        targets * np.log(probs + eps) +
        (1 - targets) * np.log(1 - probs + eps)
    )
    return probs, targets, float(bce)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    partition: str = '') -> dict:
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
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def plot_training_curves(train_losses: list, val_losses: list,
                          train_aucs: list,  val_aucs: list,
                          output_dir: Path,
                          stopped_epoch: int = None):
    """Loss and AUC curves over training epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(train_losses) + 1)

    # Loss panel
    axes[0].plot(epochs, train_losses, color='steelblue', lw=1.5,
                 label='Train loss')
    axes[0].plot(epochs, val_losses,   color='tomato', lw=1.5,
                 linestyle='--', label='Val loss')
    if stopped_epoch:
        axes[0].axvline(stopped_epoch, color='gray', linestyle=':',
                        lw=1.5, label=f'Early stop (ep {stopped_epoch})')
    gap = abs(train_losses[-1] - val_losses[-1]) if train_losses else 0
    axes[0].text(0.63, 0.88, f'Final loss gap: {gap:.3f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('BCE Loss', fontsize=11)
    axes[0].set_title('PureGCN (Fixed) — Loss Curves',
                      fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # AUC panel
    axes[1].plot(epochs, train_aucs, color='steelblue', lw=1.5,
                 label='Train AUC')
    axes[1].plot(epochs, val_aucs,   color='tomato', lw=1.5,
                 linestyle='--', label='Val AUC')
    if stopped_epoch:
        axes[1].axvline(stopped_epoch, color='gray', linestyle=':',
                        lw=1.5, label=f'Early stop (ep {stopped_epoch})')
    axes[1].set_ylim(0.4, 1.05)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('AUC', fontsize=11)
    axes[1].set_title('PureGCN (Fixed) — AUC Curves',
                      fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve_PureGCN_fixed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ loss_curve_PureGCN_fixed.png')


def plot_roc(y_val: np.ndarray, y_prob_val: np.ndarray,
             y_test: np.ndarray, y_prob_test: np.ndarray,
             optimal_threshold: float,
             output_dir: Path):
    """
    ROC curves for val and test, with the optimal threshold marked.
    The vertical dashed line shows where Youden's J is maximised on val.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for y_true, y_prob, label, color, ls in [
        (y_val,  y_prob_val,  'Validation', 'steelblue', '-'),
        (y_test, y_prob_test, 'Test',       'tomato',    '--'),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                label=f'{label} (AUC = {auc:.3f})')

    # Mark the operating point on val curve at optimal threshold
    fpr_val, tpr_val, thr_val = roc_curve(y_val, y_prob_val)
    idx = np.argmin(np.abs(thr_val - optimal_threshold))
    ax.scatter(fpr_val[idx], tpr_val[idx],
               color='steelblue', s=120, zorder=5,
               label=f'Val op. point (thr={optimal_threshold:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('PureGCN (Fixed) — ROC Curves\n'
                 '(dot = Youden J threshold on val)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_puregcn_fixed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ roc_puregcn_fixed.png')


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                   partition: str, output_dir: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Greens', ax=ax,
                    xticklabels=['Control', 'Epilepsy'],
                    yticklabels=['Control', 'Epilepsy'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'PureGCN (Fixed) — {partition} ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_puregcn_fixed_{partition.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting(results: dict, output_dir: Path):
    """Train / Val / Test AUC bar chart — primary generalisation diagnostic."""
    partitions = ['train', 'val', 'test']
    colors     = {'train': 'steelblue', 'val': 'darkorange', 'test': 'tomato'}
    x, width   = np.arange(1), 0.25

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, p in enumerate(partitions):
        auc = results.get(p, {}).get('auc', 0.0)
        ax.bar(x + (i - 1) * width, [auc], width, label=p.capitalize(),
               color=colors[p], alpha=0.85, edgecolor='black')

    tr  = results.get('train', {}).get('auc', 0)
    te  = results.get('test',  {}).get('auc', 0)
    gap = tr - te
    col = 'red' if abs(gap) > 0.10 else 'black'
    ax.text(0, max(tr, te) + 0.05,
            f'Train-Test Δ={gap:.2f}',
            ha='center', fontsize=11, fontweight='bold', color=col)
    ax.set_xticks([0])
    ax.set_xticklabels(['PureGCN (Fixed)'], fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('PureGCN Fixed — Train / Val / Test AUC',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_puregcn_fixed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ overfitting_puregcn_fixed.png')


def plot_per_patient(y_true: np.ndarray, y_prob: np.ndarray,
                     patient_ids: np.ndarray,
                     output_dir: Path):
    """Per-patient test AUC — reveals systematic failure on specific patients."""
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
    colors   = ['tomato' if a < 0.5 else 'seagreen' for a in pat_aucs[sort_idx]]

    fig, ax = plt.subplots(figsize=(max(10, len(pat_aucs) * 0.4), 5))
    ax.barh(range(len(pat_aucs)), pat_aucs[sort_idx],
            color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5, label='Chance')
    ax.axvline(float(np.mean(pat_aucs)), color='navy', linestyle='-',
               lw=2, label=f'Mean = {np.mean(pat_aucs):.3f}')
    ax.set_yticks(range(len(pat_aucs)))
    ax.set_yticklabels(np.array(patients)[sort_idx], fontsize=7)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'PureGCN Fixed — Per-Patient Test AUC\n'
                 f'({(pat_aucs < 0.5).sum()} below chance)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_patient_puregcn_fixed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ per_patient_puregcn_fixed.png')


def plot_version_comparison(fixed_results: dict,
                             original_json: Path,
                             sup_json: Path,
                             output_dir: Path):
    """
    Ablation bar chart: PureGCN original vs PureGCN fixed vs SmallGCN.
    Shows test AUC only — the central ablation result for the thesis.
    """
    models = {}

    if original_json and original_json.exists():
        with open(original_json) as f:
            orig = json.load(f)
        if orig.get('test'):
            models['PureGCN\n(original)'] = orig['test'].get('auc', 0)

    models['PureGCN\n(fixed)'] = fixed_results.get('test', {}).get('auc', 0)

    if sup_json and sup_json.exists():
        with open(sup_json) as f:
            sup = json.load(f)
        if sup.get('test'):
            models['SmallGCN\n(hand-crafted)'] = sup['test'].get('auc', 0)

    if len(models) < 2:
        return

    colors = ['#888780', '#1D9E75', '#2176AE']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(models)), list(models.values()),
                  color=colors[:len(models)],
                  edgecolor='black', alpha=0.85)

    for bar, val in zip(bars, models.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(list(models.keys()), fontsize=11)
    ax.set_ylabel('Test AUC', fontsize=12)
    ax.set_ylim(0, 1.10)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1.5, label='Chance')
    ax.set_title('Ablation: Raw CNN vs Hand-Crafted Features\n'
                 '(test AUC, TUH dataset)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_versions.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ comparison_versions.png')


def plot_sensitivity_specificity_tradeoff(y_val: np.ndarray,
                                           y_prob_val: np.ndarray,
                                           optimal_threshold: float,
                                           output_dir: Path):
    """
    Sensitivity and specificity vs threshold curve for val set.
    Shows where Youden's J is maximised and why the chosen threshold
    is better than the naive 0.5 default.
    """
    fpr, tpr, thresholds = roc_curve(y_val, y_prob_val)
    sensitivities = tpr
    specificities = 1.0 - fpr

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sensitivities, color='tomato', lw=2,
            label='Sensitivity (TPR)')
    ax.plot(thresholds, specificities, color='steelblue', lw=2,
            label='Specificity (1-FPR)')
    ax.axvline(optimal_threshold, color='seagreen', linestyle='--',
               lw=2, label=f'Optimal threshold = {optimal_threshold:.3f}')
    ax.axvline(0.5, color='gray', linestyle=':', lw=1.5,
               label='Default threshold = 0.50')
    ax.set_xlabel('Classification threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title('PureGCN Fixed — Sensitivity/Specificity vs Threshold\n'
                 '(val set, Youden J maximised at green line)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_tradeoff_puregcn_fixed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ threshold_tradeoff_puregcn_fixed.png')


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 5b — Fixed PureGCN with optimal threshold (TUH)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/pure_gcn_fixed')
    parser.add_argument('--original_json', default=None,
                        help='Path to results_pure_gcn.json from step 5 '
                             '(original collapsed version)')
    parser.add_argument('--sup_json',      default=None,
                        help='Path to results_gcn.json from step 5 (SmallGCN)')
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold_pct', type=float, default=50.0,
                        help='DTF edge percentile threshold. Default 50 '
                             'for TUH resting-state (step 5 used 70).')
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--accum_steps',   type=int,   default=8,
                        help='Gradient accumulation steps (default 8). '
                             'Effective mini-batch size for CNN weight updates.')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5b — FIXED PureGCN  (TUH dataset)')
    print('=' * 65)
    print(f'Device        : {device}')
    print(f'Epochs        : {args.epochs}   LR: {args.lr}   '
          f'Hidden: {args.hidden}')
    print(f'Dropout       : {args.dropout}   Patience: {args.patience}')
    print(f'Threshold pct : {args.threshold_pct}  '
          f'(keeps top {100-args.threshold_pct:.0f}% of edges)')
    print(f'Accum steps   : {args.accum_steps}  '
          f'(effective mini-batch size for CNN)')
    print()
    print('FIXES APPLIED:')
    print('  1. InstanceNorm1d instead of BatchNorm1d')
    print('     → stable normalisation with 22 channels per forward pass')
    print('  2. Per-channel z-score normalisation before CNN')
    print('     → removes patient-specific amplitude scale')
    print(f'  3. Gradient accumulation over {args.accum_steps} graphs')
    print('     → stable CNN weight updates')
    print("  4. Youden's J threshold on val set")
    print('     → fixes sensitivity bias from default threshold=0.5')
    print('=' * 65)

    # ── Load features ──────────────────────────────────────────────────
    print('\nLoading features...')
    data        = np.load(args.featfile, allow_pickle=True)
    adj_dtf     = data['adj_dtf'].astype(np.float32)
    y           = data['y'].astype(np.int64)
    splits      = data['splits']
    patient_ids = data['patient_ids']

    if 'raw_epochs' not in data:
        print('[ERROR] raw_epochs not found in features_all.npz')
        print('  Re-run step 3 to include raw_epochs in the output.')
        return

    raw_epochs = data['raw_epochs'].astype(np.float32)

    train_mask = splits == 'train'
    val_mask   = splits == 'val'
    test_mask  = splits == 'test'

    y_train  = y[train_mask]
    y_val    = y[val_mask]
    y_test   = y[test_mask]
    pat_test = patient_ids[test_mask]

    majority_b = max((y == 0).sum(), (y == 1).sum()) / len(y)
    n_neg      = int((y_train == 0).sum())
    n_pos      = int((y_train == 1).sum())
    pos_weight = n_neg / (n_pos + 1e-12)

    print(f'Dataset: {len(y):,} epochs total')
    print(f'  Train : {train_mask.sum():,}  Val: {val_mask.sum():,}  '
          f'Test: {test_mask.sum():,}')
    print(f'  Train balance — epi: {n_pos}, ctrl: {n_neg}  '
          f'(pos_weight={pos_weight:.3f})')
    print(f'  Majority-class baseline: {majority_b * 100:.1f}%\n')

    # ── DTF threshold from training adjacency only ─────────────────────
    print('Computing DTF threshold from training adjacency...')
    threshold = compute_threshold(adj_dtf[train_mask], args.threshold_pct)
    print(f'  Threshold (p{args.threshold_pct:.0f}): {threshold:.4f}')

    sample_adj = adj_dtf[train_mask][:100]
    mean_edges = np.mean(
        [(a > threshold).sum() - N_CHANNELS for a in sample_adj]
    )
    print(f'  Mean edges per graph: {mean_edges:.1f} '
          f'/ {N_CHANNELS * (N_CHANNELS - 1)} possible\n')

    # ── Build graphs ───────────────────────────────────────────────────
    print('Building graphs (raw EEG input)...')
    g_train = build_graphs_raw(
        raw_epochs[train_mask], adj_dtf[train_mask], threshold)
    g_val   = build_graphs_raw(
        raw_epochs[val_mask],   adj_dtf[val_mask],   threshold)
    g_test  = build_graphs_raw(
        raw_epochs[test_mask],  adj_dtf[test_mask],  threshold)
    print(f'  Graphs — train: {len(g_train)}, '
          f'val: {len(g_val)}, test: {len(g_test)}\n')

    # ── Model ──────────────────────────────────────────────────────────
    model    = FixedPureGCN(n_samples=N_SAMPLES, node_dim=NODE_DIM,
                             hidden=args.hidden,
                             dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'FixedPureGCN parameters: {n_params:,}')

    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5,
                                  verbose=False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    # ── Training loop with early stopping on val loss ──────────────────
    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    best_val_loss  = np.inf
    best_state     = None
    patience_cnt   = 0
    stopped_epoch  = None

    print(f'\nTraining (max {args.epochs} epochs, '
          f'patience={args.patience}, '
          f'accum={args.accum_steps})...')

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch_accum(
            model, optimiser, criterion,
            g_train, y_train, device,
            accum_steps=args.accum_steps,
        )

        tr_probs,  tr_targets,  _        = evaluate_graphs(
            model, g_train, y_train, device)
        val_probs, val_targets, val_loss = evaluate_graphs(
            model, g_val, y_val, device)

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
                  f'TrAUC={tr_auc:.3f}  ValAUC={val_auc:.3f}'
                  f'  Gap={gap:.3f}{flag}')

            if ep >= 5 and val_auc <= 0.51:
                print(f'  [WARN] Val AUC ≤ 0.51 at epoch {ep} — '
                      f'model may still be collapsing.')

        # Early stopping on val LOSS (not AUC — avoids leakage)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone()
                             for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                stopped_epoch = ep
                print(f'\n  Early stop at epoch {ep}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'  Best checkpoint restored (val loss = {best_val_loss:.4f})')

    plot_training_curves(train_losses, val_losses, train_aucs, val_aucs,
                         output_dir, stopped_epoch)

    # ══════════════════════════════════════════════════════════════
    # THRESHOLD SELECTION — val set only, no leakage
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 55)
    print("THRESHOLD SELECTION (Youden's J on val set)")
    print('─' * 55)

    val_probs_final, val_targets_final, _ = evaluate_graphs(
        model, g_val, y_val, device)

    optimal_threshold, opt_val_sens, opt_val_spec = find_optimal_threshold(
        val_targets_final, val_probs_final
    )

    plot_sensitivity_specificity_tradeoff(
        val_targets_final, val_probs_final,
        optimal_threshold, output_dir
    )

    # ── Final evaluation at optimal threshold ──────────────────────────
    print('\n' + '─' * 55)
    print(f'FINAL EVALUATION  (threshold = {optimal_threshold:.4f})')
    print('─' * 55)

    results   = {}
    all_probs = {}

    for partition, graphs, labels in [
        ('train', g_train, y_train),
        ('val',   g_val,   y_val),
        ('test',  g_test,  y_test),
    ]:
        probs, targets, _ = evaluate_graphs(model, graphs, labels, device)
        # Apply optimal threshold (not 0.5)
        preds = (probs >= optimal_threshold).astype(np.int64)
        m     = compute_metrics(targets, preds, probs, partition)
        results[partition]   = m
        all_probs[partition] = (probs, preds, targets)

        print(f'  {partition:5s} | AUC={m.get("auc",0):.3f}  '
              f'F1={m.get("f1",0):.3f}  '
              f'Sens={m.get("sensitivity",0):.3f}  '
              f'Spec={m.get("specificity",0):.3f}  '
              f'MCC={m.get("mcc",0):.3f}')

    tr_auc = results['train'].get('auc', 0)
    te_auc = results['test'].get('auc', 0)
    gap    = tr_auc - te_auc
    flag   = ' ⚠ OVERFIT' if gap > 0.10 else ' ✓ OK'
    print(f'\n  Train-Test AUC gap: {gap:.3f}{flag}')
    print(f'  NOTE: AUC is threshold-independent — identical to before.')
    print(f'  Sensitivity/Specificity now balanced via Youden J threshold.')

    # ── Plots ──────────────────────────────────────────────────────────
    val_probs_arr,  val_preds_arr,  val_tgts  = all_probs['val']
    test_probs_arr, test_preds_arr, test_tgts = all_probs['test']

    plot_roc(y_val, val_probs_arr, y_test, test_probs_arr,
             optimal_threshold, output_dir)

    for partition, (probs, preds, targets) in all_probs.items():
        plot_confusion(targets, preds, partition, output_dir)

    plot_overfitting(results, output_dir)
    plot_per_patient(y_test, test_probs_arr, pat_test, output_dir)

    orig_path = Path(args.original_json) if args.original_json else None
    sup_path  = Path(args.sup_json)      if args.sup_json      else None
    plot_version_comparison(results, orig_path, sup_path, output_dir)

    # ── Save results ───────────────────────────────────────────────────
    out = {
        'model': 'PureGCN_Fixed',
        'fixes': [
            'InstanceNorm1d instead of BatchNorm1d',
            'Per-channel z-score normalisation before CNN',
            f'Gradient accumulation over {args.accum_steps} graphs',
            f'DTF threshold percentile {args.threshold_pct} (step 5 used 70)',
            "Youden's J threshold selection on validation set",
        ],
        'hyperparameters':    vars(args),
        'optimal_threshold':  float(optimal_threshold),
        'threshold_method':   "Youden's J on validation set",
        'val_sens_at_threshold': float(opt_val_sens),
        'val_spec_at_threshold': float(opt_val_spec),
        'train_test_gap':     float(gap),
        'stopped_epoch':      stopped_epoch or args.epochs,
        'n_params':           n_params,
        **results,
    }
    out_path = output_dir / 'results_pure_gcn_fixed.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n  ✓ Results → {out_path}')

    # ── Final summary ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('STEP 5b COMPLETE — FINAL SUMMARY')
    print('=' * 65)
    print(f'{"Partition":8s} | {"AUC":>6} {"F1":>6} '
          f'{"Sens":>6} {"Spec":>6} {"MCC":>6}')
    print('-' * 45)
    for p in ['train', 'val', 'test']:
        m = results.get(p, {})
        print(f'{p:8s} | {m.get("auc",0):6.3f} {m.get("f1",0):6.3f} '
              f'{m.get("sensitivity",0):6.3f} {m.get("specificity",0):6.3f} '
              f'{m.get("mcc",0):6.3f}')

    print(f'\nOptimal threshold : {optimal_threshold:.4f}  '
          f'(Youden J on val)')
    print(f'Train-Test gap    : {gap:.3f}{flag}')
    print()
    print('THESIS ABLATION TABLE (test AUC):')
    print(f'  PureGCN original (collapsed) : 0.500')
    print(f'  PureGCN fixed                : {te_auc:.3f}')
    if orig_path and orig_path.exists():
        pass
    if sup_path and sup_path.exists():
        with open(sup_path) as f:
            sup = json.load(f)
        small_auc = sup.get('test', {}).get('auc', 0)
        print(f'  SmallGCN (hand-crafted)      : {small_auc:.3f}')
        gap_to_small = small_auc - te_auc
        if gap_to_small > 0.05:
            print(f'\n  SmallGCN outperforms PureGCN fixed by {gap_to_small:.3f} AUC.')
            print('  Hand-crafted features are more effective than CNN-learned')
            print('  embeddings even with 52k training graphs and proper normalisation.')
        else:
            print(f'\n  PureGCN fixed is within {gap_to_small:.3f} AUC of SmallGCN.')
            print('  CNN-learned features are competitive with hand-crafted ones')
            print('  when training is properly regularised.')
    print('=' * 65)


if __name__ == '__main__':
    main()
