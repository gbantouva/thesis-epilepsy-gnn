"""
Step 5b — PureGCN (Fixed, TUH Dataset)
=======================================
PURPOSE:
  Standalone ablation script that re-runs ONLY PureGCN with two critical
  fixes for the collapse observed in step 5:

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

  ADDITIONALLY — per-channel z-score normalisation is kept inside the
    encoder forward pass. This removes patient-specific amplitude scale
    before the CNN, which is the most important single fix for
    cross-patient generalisation.

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
    This is the thesis argument.

EVALUATION:
  Same fixed 60-20-20 split as step 5. Same threshold. Same GCN head.
  The only difference from SmallGCN is the node feature source.

Outputs (saved to --outputdir):
  results_pure_gcn_fixed.json
  loss_curve_PureGCN_fixed.png
  roc_puregcn_fixed.png
  cm_puregcn_fixed_{partition}.png
  overfitting_puregcn_fixed.png
  per_patient_puregcn_fixed.png
  comparison_versions.png          original vs fixed PureGCN

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
from sklearn.preprocessing import StandardScaler

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

    NOTE: For TUH resting-state EEG, percentile=50 (keep top 50% edges)
    is recommended over the 70% used in TUC. TUH connectivity matrices
    are sparser on average — keeping only top 30% risks near-empty graphs
    for control patients with weak resting connectivity.
    """
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    vals = np.concatenate(
        [adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))]
    )
    return float(np.percentile(vals, percentile))


def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """A_hat = D^{-1/2} (A + I) D^{-1/2}"""
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
# FIXED MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class FixedRawChannelEncoder(nn.Module):
    """
    FIXED 1D-CNN encoder for raw EEG per channel.

    KEY FIXES vs original step 5 RawChannelEncoder:

    FIX 1 — InstanceNorm1d instead of BatchNorm1d
      BatchNorm requires stable cross-sample statistics. When processing
      one graph at a time (22 channels per forward pass), BatchNorm1d
      receives n=22 — too few for stable estimates. All outputs collapse
      to similar embeddings, giving the GCN no discriminative signal.

      InstanceNorm1d normalises within each channel independently.
      No cross-sample statistics needed. Correct for this use case.
      affine=True preserves learnable scale and shift parameters.

    FIX 2 — Per-channel z-score normalisation BEFORE CNN (kept from step 5b TUC)
      Removes patient-specific amplitude scale before the CNN sees the signal.
      Forces the CNN to learn shape-based temporal features (oscillations,
      transients) rather than absolute amplitude. Critical for cross-patient
      generalisation.

    FIX 3 — Smaller kernel, stride, output (same as TUC step 5b)
      Conv1d(1, 8, kernel=64, stride=16) → InstanceNorm → ReLU → Pool(4)
      → Linear(32 → node_dim)
      ~560 CNN parameters. Deliberately small to avoid memorisation.

    Input  : (22, 1000) — raw EEG epoch, one row per channel
    Output : (22, node_dim)
    """
    def __init__(self, n_samples: int = N_SAMPLES,
                 node_dim: int = NODE_DIM,
                 dropout: float = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=64, stride=16, padding=32),
            nn.InstanceNorm1d(8, affine=True),   # FIX 1: was BatchNorm1d
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(4),              # → (8, 4)
        )
        self.proj  = nn.Linear(8 * 4, node_dim)  # 32 → 17
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        FIX 2: Per-channel z-score normalisation BEFORE CNN.
        Each of the 22 channels normalised independently.
        """
        mean   = x_raw.mean(dim=1, keepdim=True)          # (22, 1)
        std    = x_raw.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_norm = (x_raw - mean) / std                     # (22, 1000)

        n_ch = x_norm.shape[0]
        h    = self.conv(x_norm.unsqueeze(1))              # (22, 8, 4)
        h    = self.drop2(h.view(n_ch, -1))                # (22, 32)
        return self.proj(h)                                # (22, node_dim)


class GCNLayer(nn.Module):
    """Single GCN layer: H' = ReLU( A_hat @ H @ W )"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a @ x))


class FixedPureGCN(nn.Module):
    """
    FIXED PureGCN — raw EEG → FixedRawChannelEncoder → GCN → classifier.

    GCN head identical to SmallGCN in step 5 for fair ablation.
    Only the node feature source changes.

    Architecture:
        FixedRawChannelEncoder  (22, 1000) → (22, NODE_DIM)
        GCNLayer(NODE_DIM → hidden)
        Dropout
        GCNLayer(hidden → hidden)
        GlobalMeanPool  → (hidden,)
        Linear(hidden → 16) → ReLU → Dropout
        Linear(16 → 1)
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
        h        = h.mean(dim=0, keepdim=True)
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

    WHY: Updating weights after every single graph produces very noisy
    gradients. The CNN encoder needs to see diverse inputs within each
    update step to learn general temporal features rather than fitting
    individual graphs. Accumulating gradients over `accum_steps` graphs
    before each optimiser step simulates a mini-batch of size accum_steps
    without requiring memory for a full batch.

    EFFECT: Effective mini-batch size = accum_steps (default 8).
    Gradients are averaged over the accumulated steps before the update.
    This is mathematically equivalent to computing the mean loss over
    accum_steps graphs.

    Parameters
    ----------
    accum_steps : int
        Number of graphs to accumulate before each weight update.
        Higher = more stable gradients, slower updates.
        Recommended: 8-16 for TUH. Too high (>32) may slow convergence.
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
        # Divide loss by accum_steps so accumulated gradient = mean gradient
        loss  = criterion(logit.unsqueeze(0), label) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps   # undo scaling for logging

        # Update weights every accum_steps graphs
        if (step + 1) % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            optimiser.zero_grad()

    # Final update for remaining graphs
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


def compute_metrics(y_true, y_pred, y_prob,
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

def plot_training_curves(train_losses, val_losses,
                          train_aucs, val_aucs,
                          output_dir: Path, stopped_epoch=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(train_losses) + 1)

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


def plot_roc(y_val, y_prob_val, y_test, y_prob_test, output_dir: Path):
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
    ax.set_title('PureGCN (Fixed) — ROC Curves',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_puregcn_fixed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ roc_puregcn_fixed.png')


def plot_confusion(y_true, y_pred, partition: str, output_dir: Path):
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
    col = 'red' if gap > 0.10 else 'black'
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


def plot_per_patient(y_true, y_prob, patient_ids, output_dir: Path):
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
    fig, ax  = plt.subplots(figsize=(max(10, len(pat_aucs) * 0.4), 5))
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
    Three-model comparison: PureGCN original vs PureGCN fixed vs SmallGCN.
    Shows test AUC for each. This is the ablation table figure.
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


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 5b — Fixed PureGCN (TUH dataset)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/pure_gcn_fixed')
    parser.add_argument('--original_json', default=None,
                        help='Path to results_pure_gcn.json from step 5')
    parser.add_argument('--sup_json',      default=None,
                        help='Path to results_gcn.json from step 5 (SmallGCN)')
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--lr',            type=float, default=0.001)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold_pct', type=float, default=50.0,
                        help='DTF percentile threshold (default 50 for TUH, '
                             'vs 70 in step 5 — see docstring)')
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--accum_steps',   type=int,   default=8,
                        help='Gradient accumulation steps (default 8)')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5b — FIXED PureGCN  (TUH dataset)')
    print('=' * 65)
    print(f'Device        : {device}')
    print(f'Epochs        : {args.epochs}  LR: {args.lr}  Hidden: {args.hidden}')
    print(f'Dropout       : {args.dropout}  Patience: {args.patience}')
    print(f'Threshold pct : {args.threshold_pct}  '
          f'(keeps top {100-args.threshold_pct:.0f}% edges)')
    print(f'Accum steps   : {args.accum_steps}  '
          f'(effective mini-batch size)')
    print()
    print('FIXES APPLIED:')
    print('  1. InstanceNorm1d instead of BatchNorm1d in CNN encoder')
    print('     (stable normalisation with 22 channels per forward pass)')
    print('  2. Per-channel z-score normalisation before CNN')
    print('     (removes patient-specific amplitude scale)')
    print('  3. Gradient accumulation over', args.accum_steps, 'graphs')
    print('     (more stable gradients for CNN weight updates)')
    print('=' * 65)

    # ── Load features ──────────────────────────────────────────────────
    data        = np.load(args.featfile, allow_pickle=True)
    adj_dtf     = data['adj_dtf'].astype(np.float32)        # (N, 22, 22)
    y           = data['y'].astype(np.int64)
    splits      = data['splits']
    patient_ids = data['patient_ids']

    if 'raw_epochs' not in data:
        print('[ERROR] raw_epochs not found in features_all.npz')
        print('  Re-run step 3 to include raw_epochs in the output.')
        return

    raw_epochs = data['raw_epochs'].astype(np.float32)      # (N, 22, 1000)

    train_mask = splits == 'train'
    val_mask   = splits == 'val'
    test_mask  = splits == 'test'

    y_train    = y[train_mask]
    y_val      = y[val_mask]
    y_test     = y[test_mask]
    pat_test   = patient_ids[test_mask]

    majority_b = max((y == 0).sum(), (y == 1).sum()) / len(y)
    n_neg      = int((y_train == 0).sum())
    n_pos      = int((y_train == 1).sum())
    pos_weight = n_neg / (n_pos + 1e-12)

    print(f'\nDataset: {len(y):,} epochs total')
    print(f'  Train : {train_mask.sum():,}  Val: {val_mask.sum():,}  '
          f'Test: {test_mask.sum():,}')
    print(f'  pos_weight: {pos_weight:.3f}')
    print(f'  Majority-class baseline: {majority_b * 100:.1f}%\n')

    # ── Threshold from training adjacency only ─────────────────────────
    print('Computing DTF threshold from training adjacency...')
    threshold = compute_threshold(adj_dtf[train_mask], args.threshold_pct)
    print(f'  Threshold (p{args.threshold_pct:.0f}): {threshold:.4f}')

    # Sanity check: how many edges survive on average?
    sample_adj  = adj_dtf[train_mask][:100]
    mean_edges  = np.mean([(a > threshold).sum() - N_CHANNELS
                           for a in sample_adj])
    print(f'  Mean edges per graph after thresholding: {mean_edges:.1f} '
          f'/ {N_CHANNELS*(N_CHANNELS-1)} possible\n')

    # ── Build graphs ───────────────────────────────────────────────────
    print('Building graphs (raw EEG)...')
    g_train = build_graphs_raw(
        raw_epochs[train_mask], adj_dtf[train_mask], threshold)
    g_val   = build_graphs_raw(
        raw_epochs[val_mask],   adj_dtf[val_mask],   threshold)
    g_test  = build_graphs_raw(
        raw_epochs[test_mask],  adj_dtf[test_mask],  threshold)
    print(f'  Graphs — train: {len(g_train)}, '
          f'val: {len(g_val)}, test: {len(g_test)}')

    # ── Model ──────────────────────────────────────────────────────────
    model    = FixedPureGCN(n_samples=N_SAMPLES, node_dim=NODE_DIM,
                             hidden=args.hidden, dropout=args.dropout
                             ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n  FixedPureGCN parameters: {n_params:,}')

    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5,
                                  verbose=False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    # ── Training loop ──────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    best_val_loss  = np.inf
    best_state     = None
    patience_cnt   = 0
    stopped_epoch  = None

    print(f'\n  Training with gradient accumulation '
          f'(accum_steps={args.accum_steps})...')

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch_accum(
            model, optimiser, criterion,
            g_train, y_train, device,
            accum_steps=args.accum_steps,
        )

        tr_probs, _, tr_targets, _ = evaluate_graphs(
            model, g_train, y_train, device)
        val_probs, val_preds, val_targets, val_loss = evaluate_graphs(
            model, g_val, y_val, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        tr_auc  = float(roc_auc_score(tr_targets, tr_probs)) \
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

            # Early collapse detection
            if ep >= 5 and val_auc <= 0.51:
                print(f'\n  [WARN] Val AUC ≤ 0.51 at epoch {ep} — '
                      f'model may be collapsing.')
                print(f'  If this persists, try reducing --accum_steps '
                      f'or lowering --lr.')

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
        print(f'  Best checkpoint restored '
              f'(val loss = {best_val_loss:.4f})')

    # ── Plots ──────────────────────────────────────────────────────────
    plot_training_curves(train_losses, val_losses, train_aucs, val_aucs,
                         output_dir, stopped_epoch)

    # ── Final evaluation ───────────────────────────────────────────────
    print('\n  Final evaluation...')
    results   = {}
    all_probs = {}

    for partition, graphs, labels in [
        ('train', g_train, y_train),
        ('val',   g_val,   y_val),
        ('test',  g_test,  y_test),
    ]:
        probs, preds, targets, _ = evaluate_graphs(
            model, graphs, labels, device)
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

    val_probs_arr,  val_preds_arr,  _ = all_probs['val']
    test_probs_arr, test_preds_arr, _ = all_probs['test']

    plot_roc(y_val, val_probs_arr, y_test, test_probs_arr, output_dir)
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
            f'Threshold percentile {args.threshold_pct} '
            f'(was 70 in original step 5)',
        ],
        'hyperparameters': vars(args),
        'train_test_gap':  float(gap),
        'stopped_epoch':   stopped_epoch or args.epochs,
        'n_params':        n_params,
        **results,
    }
    out_path = output_dir / 'results_pure_gcn_fixed.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n  ✓ Results → {out_path}')

    # ── Final summary ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('THESIS ABLATION TABLE (test AUC)')
    print('=' * 65)
    print(f'  SmallGCN   (hand-crafted, step 5) : '
          f'load from results_gcn.json')
    print(f'  PureGCN    (original,    step 5) : AUC = 0.500 (collapsed)')
    print(f'  PureGCN    (fixed,       step 5b): '
          f'AUC = {te_auc:.3f}')
    print()
    if te_auc > 0.55:
        print('  PureGCN fixed is now above chance.')
        print('  Compare against SmallGCN test AUC to determine')
        print('  whether hand-crafted features are still necessary.')
    else:
        print('  PureGCN fixed is still near chance.')
        print('  The CNN cannot learn cross-patient EEG features')
        print('  even with stable normalisation and gradient accumulation.')
        print('  This strongly justifies hand-crafted features.')
    print('=' * 65)


if __name__ == '__main__':
    main()
