"""
Step 5 — Supervised GCN (TUH Dataset, Fixed 60-20-20 Split)
=============================================================
Two models evaluated on the same fixed split:
  MODEL A: SmallGCN  — hand-crafted node features (22, 17)
  MODEL B: PureGCN   — raw EEG node features learned by a 1D-CNN

OVERFITTING PREVENTION:
  1. Dropout (p=0.4) in GCN layers and classifier head
  2. L2 weight decay (1e-4) in Adam
  3. Gradient clipping (max_norm=1.0)
  4. ReduceLROnPlateau (patience=10, factor=0.5)
  5. Early stopping on VAL LOSS (not val AUC — avoids leakage)
  6. Best-state restore from checkpoint with lowest val loss
  7. BCEWithLogitsLoss with pos_weight (class imbalance)
  8. Node feature StandardScaler fit on TRAIN split only

OVERFITTING DIAGNOSTICS (same plots as TUC LOPO version):
  loss_curve_<model>.png     train vs val loss + AUC per epoch
  overfitting_<model>.png    train AUC vs test AUC summary bar
  cm_<model>_test.png        confusion matrix (test set)
  roc_<model>.png            ROC: val + test curves
  results_gcn.json

Usage:
  python step5_gnn_supervised.py \\
      --featfile   F:\\features\\features_all.npz \\
      --output_dir F:\\results\\gnn_supervised \\
      --epochs 150 --lr 0.001 --hidden 32 \\
      --threshold_pct 70 --dropout 0.4 --patience 20 \\
      --baseline_json F:\\results\\baseline_ml\\results_all.json
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
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              matthews_corrcoef, precision_score,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

N_CHANNELS = 22
N_SAMPLES  = 1000
N_NODE_FEATS = 17


# ─────────────────────────────────────────────────────────────
# Graph utilities
# ─────────────────────────────────────────────────────────────

def compute_threshold(adj_dtf_train: np.ndarray,
                      percentile: float = 70.0) -> float:
    """Percentile of off-diagonal DTF values — computed on TRAIN only."""
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    vals = np.concatenate([adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))])
    return float(np.percentile(vals, percentile))


def normalize_adjacency(adj: np.ndarray) -> torch.Tensor:
    """Symmetric normalisation: Â = D^{-1/2}(A+I)D^{-1/2}."""
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs_features(node_feats: np.ndarray,
                           adj_dtf:    np.ndarray,
                           threshold:  float) -> list:
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        ahat = normalize_adjacency(adj)
        x    = torch.tensor(node_feats[i], dtype=torch.float32)  # (22, 17)
        graphs.append((x, ahat))
    return graphs


def build_graphs_raw(raw_epochs: np.ndarray,
                     adj_dtf:    np.ndarray,
                     threshold:  float) -> list:
    graphs = []
    for i in range(len(raw_epochs)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        ahat = normalize_adjacency(adj)
        x    = torch.tensor(raw_epochs[i], dtype=torch.float32)  # (22, 1000)
        graphs.append((x, ahat))
    return graphs


def scale_node_features(graphs_train: list, graphs_test: list,
                         graphs_val:  list) -> tuple:
    """Fit StandardScaler on train node features; apply to val and test."""
    all_train_x = np.concatenate([g[0].numpy() for g in graphs_train], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_train_x)
    def apply(graphs):
        return [(torch.tensor(scaler.transform(x.numpy()),
                              dtype=torch.float32), a)
                for x, a in graphs]
    return apply(graphs_train), apply(graphs_val), apply(graphs_test)


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, ahat: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(ahat @ x))


class SmallGCN(nn.Module):
    """Feature-based GCN. Input: (22, 17) node features."""
    def __init__(self, in_dim=N_NODE_FEATS, hidden=32, dropout=0.4):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x, ahat):
        h = self.gcn1(x, ahat)
        h = self.drop(h)
        h = self.gcn2(h, ahat)
        h = h.mean(dim=0, keepdim=True)  # global mean pool
        return self.head(h).squeeze()


class RawChannelEncoder(nn.Module):
    """1D-CNN per channel: (22, 1000) → (22, node_dim)."""
    def __init__(self, n_samples=N_SAMPLES, node_dim=N_NODE_FEATS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8,  kernel_size=32, stride=8,  padding=16), nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=16, stride=4,  padding=8),  nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = nn.Linear(16 * 8, node_dim)

    def forward(self, x_raw):   # (22, 1000)
        n_ch = x_raw.shape[0]
        h = self.conv(x_raw.unsqueeze(1))     # (22, 16, 8)
        h = h.view(n_ch, -1)                  # (22, 128)
        return self.proj(h)                   # (22, node_dim)


class PureGCN(nn.Module):
    """Raw-signal GCN. Input: (22, 1000) raw EEG."""
    def __init__(self, n_samples=N_SAMPLES, node_dim=N_NODE_FEATS,
                 hidden=32, dropout=0.4):
        super().__init__()
        self.encoder = RawChannelEncoder(n_samples, node_dim)
        self.gcn1 = GCNLayer(node_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x_raw, ahat):
        node_emb = self.encoder(x_raw)        # (22, node_dim)
        h = self.gcn1(node_emb, ahat)
        h = self.drop(h)
        h = self.gcn2(h, ahat)
        h = h.mean(dim=0, keepdim=True)
        return self.head(h).squeeze()


# ─────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, optimiser, criterion, graphs, labels, device):
    model.train()
    total_loss = 0.0
    for i in np.random.permutation(len(graphs)):
        x, a    = graphs[i]
        x, a    = x.to(device), a.to(device)
        optimiser.zero_grad()
        logit   = model(x, a)
        label   = torch.tensor(float(labels[i]), device=device).unsqueeze(0)
        loss    = criterion(logit.unsqueeze(0), label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(graphs)


@torch.no_grad()
def evaluate_graphs(model, graphs, labels, device):
    model.eval()
    logits  = []
    targets = []
    for i in range(len(graphs)):
        x, a = graphs[i]
        logit = model(x.to(device), a.to(device))
        logits.append(logit.cpu().item())
        targets.append(int(labels[i]))
    logits  = np.array(logits,  dtype=np.float32)
    targets = np.array(targets, dtype=np.int64)
    probs   = 1.0 / (1.0 + np.exp(-logits))
    preds   = (probs > 0.5).astype(np.int64)
    bce     = float(-np.mean(targets * np.log(probs + 1e-7)
                             + (1 - targets) * np.log(1 - probs + 1e-7)))
    return probs, preds, targets, bce


def compute_metrics(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return None
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_total    = len(y_true)
    majority_n = max((y_true == 0).sum(), (y_true == 1).sum())
    return dict(
        accuracy          = float(accuracy_score(y_true, y_pred)),
        majority_baseline = float(majority_n / n_total),
        auc               = float(roc_auc_score(y_true, y_prob)),
        f1                = float(f1_score(y_true, y_pred, zero_division=0)),
        sensitivity       = float(tp / (tp + fn + 1e-12)),
        specificity       = float(tn / (tn + fp + 1e-12)),
        precision         = float(precision_score(y_true, y_pred, zero_division=0)),
        mcc               = float(matthews_corrcoef(y_true, y_pred)),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_loss_and_auc(train_losses, val_losses, train_aucs, val_aucs,
                      model_tag, output_dir):
    """Two-panel: loss curves (left) + AUC curves (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(train_losses, color='royalblue',  lw=1.5, label='Train loss')
    axes[0].plot(val_losses,   color='tomato',     lw=1.5, linestyle='--',
                 label='Val loss')
    gap = abs(train_losses[-1] - val_losses[-1])
    axes[0].text(0.63, 0.88, f'Final gap: {gap:.4f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('BCE Loss', fontsize=11)
    axes[0].set_title(f'{model_tag} — Loss Curves', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_aucs, color='royalblue', lw=1.5, label='Train AUC')
    axes[1].plot(val_aucs,   color='tomato',    lw=1.5, linestyle='--',
                 label='Val AUC')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('AUC', fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f'{model_tag} — AUC Curves', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'loss_curve_{model_tag.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, model_tag, output_dir, cmap='Blues'):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    xticklabels=['Control', 'Epilepsy'],
                    yticklabels=['Control', 'Epilepsy'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True',      fontsize=11)
        ax.set_title(f'{model_tag} — Test CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_{model_tag.lower().replace(" ", "_")}_test.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(val_metrics, test_metrics, fpr_va, tpr_va, fpr_te, tpr_te,
             model_tag, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_va, tpr_va, lw=2, color='steelblue',
            label=f'Val  AUC={val_metrics["auc"]:.3f}')
    ax.plot(fpr_te, tpr_te, lw=2, color='tomato',
            label=f'Test AUC={test_metrics["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'{model_tag} — ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_tag.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting(all_results, output_dir):
    """Train / Val / Test AUC bar chart per model with gap annotation."""
    model_names = list(all_results.keys())
    tr_aucs  = [all_results[m]['train_auc']         for m in model_names]
    va_aucs  = [all_results[m]['val_metrics']['auc'] for m in model_names]
    te_aucs  = [all_results[m]['test_metrics']['auc']for m in model_names]

    x, w = np.arange(len(model_names)), 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w,  tr_aucs, w, label='Train AUC', color='steelblue',
           alpha=0.85, edgecolor='black')
    ax.bar(x,      va_aucs, w, label='Val AUC',   color='seagreen',
           alpha=0.85, edgecolor='black')
    ax.bar(x + w,  te_aucs, w, label='Test AUC',  color='tomato',
           alpha=0.85, edgecolor='black')
    for i, (tr, te) in enumerate(zip(tr_aucs, te_aucs)):
        gap   = tr - te
        color = 'red' if gap > 0.10 else 'black'
        ax.text(i, max(tr, va_aucs[i], te) + 0.03, f'gap={gap:.2f}',
                ha='center', fontsize=10, fontweight='bold', color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('GCN — Overfitting: Train / Val / Test AUC',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_gcn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  overfitting_gcn.png')


def plot_model_comparison(all_results, baseline_json, output_dir):
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    models   = {}

    if baseline_json and Path(baseline_json).exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if res.get('test_metrics'):
                models[name] = res['test_metrics']

    for name, res in all_results.items():
        models[name] = res['test_metrics']

    colors = ['steelblue', 'tomato', 'seagreen', 'mediumpurple', 'darkorange']
    x      = np.arange(len(met_keys))
    width  = 0.18
    n      = len(models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, stats) in enumerate(models.items()):
        vals   = [stats[k] for k in met_keys]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score (Test set)', fontsize=12)
    ax.set_ylim(0, 1.30)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('All Models — Test Set Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  comparison_all_models.png')


# ─────────────────────────────────────────────────────────────
# Training loop (one model, fixed split)
# ─────────────────────────────────────────────────────────────

def run_training(model_name, model_factory, graphs_tr, graphs_va, graphs_te,
                 y_tr, y_va, y_te, args, output_dir, device,
                 is_feature_model=True, cmap='Blues'):

    print(f'\n{"─"*60}')
    print(f'  {model_name}')
    print(f'{"─"*60}')

    model     = model_factory().to(device)
    n_neg     = int((y_tr == 0).sum())
    n_pos     = int((y_tr == 1).sum())
    pos_weight= n_neg / (n_pos + 1e-12)
    optimiser = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    best_val_loss = np.inf
    best_state    = None
    patience_cnt  = 0

    for ep in range(args.epochs):
        tr_loss = train_one_epoch(model, optimiser, criterion,
                                  graphs_tr, y_tr, device)
        tr_probs, _, tr_tgts, _ = evaluate_graphs(model, graphs_tr, y_tr, device)
        va_probs, _, va_tgts, va_loss = evaluate_graphs(model, graphs_va, y_va, device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        tr_auc = float(roc_auc_score(tr_tgts, tr_probs)) \
                 if len(np.unique(tr_tgts)) > 1 else 0.0
        va_auc = float(roc_auc_score(va_tgts, va_probs)) \
                 if len(np.unique(va_tgts)) > 1 else 0.0
        train_aucs.append(tr_auc)
        val_aucs.append(va_auc)

        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            print(f'    Early stop at epoch {ep+1} (val loss patience)')
            break

        if (ep + 1) % 25 == 0:
            print(f'    Ep {ep+1:4d}/{args.epochs}  '
                  f'tr_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  '
                  f'tr_auc={tr_auc:.3f}  val_auc={va_auc:.3f}')

    if best_state:
        model.load_state_dict(best_state)

    # ── Final evaluation ───────────────────────────────────────
    tr_probs, tr_preds, tr_tgts, _ = evaluate_graphs(model, graphs_tr, y_tr, device)
    va_probs, va_preds, va_tgts, _ = evaluate_graphs(model, graphs_va, y_va, device)
    te_probs, te_preds, te_tgts, _ = evaluate_graphs(model, graphs_te, y_te, device)

    train_auc   = float(roc_auc_score(tr_tgts, tr_probs)) \
                  if len(np.unique(tr_tgts)) > 1 else 0.0
    val_metrics  = compute_metrics(va_tgts, va_preds, va_probs)
    test_metrics = compute_metrics(te_tgts, te_preds, te_probs)

    gap  = train_auc - test_metrics['auc']
    flag = '  ⚠ OVERFIT' if gap > 0.10 else ''
    print(f'\n  Train AUC={train_auc:.3f}  '
          f'Val AUC={val_metrics["auc"]:.3f}  '
          f'Test AUC={test_metrics["auc"]:.3f}  '
          f'Gap(Tr-Te)={gap:.3f}{flag}')
    print(f'  Test — F1={test_metrics["f1"]:.3f}  '
          f'Sens={test_metrics["sensitivity"]:.3f}  '
          f'Spec={test_metrics["specificity"]:.3f}  '
          f'MCC={test_metrics["mcc"]:.3f}')

    # ── Plots ─────────────────────────────────────────────────
    plot_loss_and_auc(train_losses, val_losses, train_aucs, val_aucs,
                      model_name, output_dir)
    plot_confusion_matrix(confusion_matrix(te_tgts, te_preds),
                          model_name, output_dir, cmap=cmap)

    fpr_va, tpr_va, _ = roc_curve(va_tgts, va_probs)
    fpr_te, tpr_te, _ = roc_curve(te_tgts, te_probs)
    plot_roc(val_metrics, test_metrics, fpr_va, tpr_va, fpr_te, tpr_te,
             model_name, output_dir)

    return dict(
        train_auc    = train_auc,
        val_metrics  = val_metrics,
        test_metrics = test_metrics,
        overfit_gap  = round(gap, 4),
        stopped_epoch= len(train_losses),
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Step 5 — Supervised GCN (TUH, fixed split)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',       required=True)
    parser.add_argument('--output_dir',     default='results/gnn_supervised')
    parser.add_argument('--epochs',         type=int,   default=150)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--hidden',         type=int,   default=32)
    parser.add_argument('--dropout',        type=float, default=0.4)
    parser.add_argument('--threshold_pct',  type=float, default=70.0)
    parser.add_argument('--patience',       type=int,   default=20)
    parser.add_argument('--baseline_json',  default=None)
    parser.add_argument('--skip_pure_gcn',  action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 5 — SUPERVISED GCN (TUH, 22 ch, fixed 60-20-20 split)')
    print('=' * 65)
    print(f'Device : {device}')
    print(f'Epochs : {args.epochs}  LR : {args.lr}  '
          f'Hidden : {args.hidden}  Dropout : {args.dropout}')
    print(f'Threshold pct : {args.threshold_pct}  '
          f'(top {100-args.threshold_pct:.0f}% edges, from TRAIN only)')
    print(f'Patience : {args.patience}  (early stop on val loss)')

    # ── Load ──────────────────────────────────────────────────
    data       = np.load(args.featfile, allow_pickle=True)
    node_feats = data['node_features'].astype(np.float32)  # (N, 22, 17)
    raw_epochs = data['raw_epochs'].astype(np.float32)     # (N, 22, 1000)
    adj_dtf    = data['adj_dtf'].astype(np.float32)        # (N, 22, 22)
    y          = data['y'].astype(np.int64)
    splits     = data['splits']

    tr = splits == 'train'
    va = splits == 'val'
    te = splits == 'test'

    print(f'\nTrain : {tr.sum():,}  Val : {va.sum():,}  Test : {te.sum():,}')

    # ── Threshold from TRAIN adjacency only ───────────────────
    threshold = compute_threshold(adj_dtf[tr], args.threshold_pct)
    print(f'DTF edge threshold (p{args.threshold_pct:.0f} of train): {threshold:.4f}')

    # ── Build graphs ──────────────────────────────────────────
    gtr_raw = build_graphs_features(node_feats[tr], adj_dtf[tr], threshold)
    gva_raw = build_graphs_features(node_feats[va], adj_dtf[va], threshold)
    gte_raw = build_graphs_features(node_feats[te], adj_dtf[te], threshold)
    gtr, gva, gte = scale_node_features(gtr_raw, gte_raw, gva_raw)
    # Note: scale_node_features returns (train, test, val) → reorder
    gtr, gva, gte = scale_node_features(gtr_raw, gva_raw, gte_raw)

    y_tr, y_va, y_te = y[tr], y[va], y[te]

    all_results = {}

    # ── SmallGCN ──────────────────────────────────────────────
    def make_small_gcn():
        return SmallGCN(in_dim=N_NODE_FEATS, hidden=args.hidden,
                        dropout=args.dropout)

    all_results['SmallGCN'] = run_training(
        'SmallGCN', make_small_gcn, gtr, gva, gte,
        y_tr, y_va, y_te, args, output_dir, device,
        is_feature_model=True, cmap='Blues',
    )

    # ── PureGCN ───────────────────────────────────────────────
    if not args.skip_pure_gcn and raw_epochs is not None:
        gtr_r = build_graphs_raw(raw_epochs[tr], adj_dtf[tr], threshold)
        gva_r = build_graphs_raw(raw_epochs[va], adj_dtf[va], threshold)
        gte_r = build_graphs_raw(raw_epochs[te], adj_dtf[te], threshold)

        def make_pure_gcn():
            return PureGCN(n_samples=N_SAMPLES, node_dim=N_NODE_FEATS,
                           hidden=args.hidden, dropout=args.dropout)

        all_results['PureGCN'] = run_training(
            'PureGCN', make_pure_gcn, gtr_r, gva_r, gte_r,
            y_tr, y_va, y_te, args, output_dir, device,
            is_feature_model=False, cmap='Greens',
        )

    # ── Overfitting + comparison plots ────────────────────────
    plot_overfitting(all_results, output_dir)
    plot_model_comparison(all_results, args.baseline_json, output_dir)

    # ── Save JSON ─────────────────────────────────────────────
    results_path = output_dir / 'results_gcn.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  ✓ Saved: {results_path}')

    print('\n' + '=' * 65)
    print('STEP 5 COMPLETE')
    print('=' * 65)
    print(f'\nNext:  python step6_ssl_gnn.py  --featfile {args.featfile}  '
          f'--sup_json {results_path}')


if __name__ == '__main__':
    main()