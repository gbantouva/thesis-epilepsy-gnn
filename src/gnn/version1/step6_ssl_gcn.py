"""
Step 6 — Self-Supervised GCN, GraphCL-style (TUH, Fixed 60-20-20 Split)
=========================================================================
PROTOCOL (leakage-free):
  1. Threshold computed from TRAIN adjacency only
  2. Node feature scaler fit on TRAIN only
  3. SSL pre-train encoder on TRAIN graphs (no labels)
  4. Phase A: linear probe — frozen encoder, 30 epochs on TRAIN labels
  5. Phase B: full fine-tune on TRAIN labels, early stop on VAL loss
  6. Final evaluation on TEST set

OVERFITTING PREVENTION:
  - Dropout (p=0.4), L2 weight decay (1e-4), gradient clipping
  - Early stopping on VAL LOSS (not val AUC — avoids test-label leakage)
  - Best-state restore from lowest val loss checkpoint
  - CosineAnnealingLR during SSL pre-training
  - ReduceLROnPlateau during fine-tuning

AUGMENTATIONS (EEG-specific):
  1. Edge dropout     (p=0.20) — mimics absent connectivity
  2. Node feature noise (σ=0.10) — mimics electrode noise
  3. Band feature mask — zero one random band (our contribution)

OVERFITTING DIAGNOSTICS:
  ssl_loss_pretrain.png     NT-Xent loss curve (train only)
  ft_curves_<phase>.png     fine-tune loss + AUC curves (train vs val)
  overfitting_ssl.png       train / val / test AUC bar chart
  cm_ssl_test.png           confusion matrix (test)
  roc_ssl.png               ROC: val + test
  comparison_all_models.png RF + SVM + GCN + SSL-GCN

Usage:
  python step6_ssl_gnn.py \\
      --featfile    F:\\features\\features_all.npz \\
      --output_dir  F:\\results\\ssl_gnn \\
      --ssl_epochs  200 --ft_epochs 100 \\
      --lr_ssl 0.001 --lr_ft 0.0005 \\
      --hidden 32 --dropout 0.4 --threshold_pct 70 \\
      --temperature 0.5 --batch_size 32 --patience 20 \\
      --baseline_json F:\\results\\baseline_ml\\results_all.json \\
      --sup_json      F:\\results\\gnn_supervised\\results_gcn.json
"""

import argparse
import copy
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              matthews_corrcoef, precision_score,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

N_CHANNELS   = 22
N_NODE_FEATS = 17
N_BAND_FEATS = 7   # TUH has 7 bands


# ─────────────────────────────────────────────────────────────
# Graph utilities (identical to step 5)
# ─────────────────────────────────────────────────────────────

def compute_threshold(adj_dtf_train, percentile=70.0):
    n    = adj_dtf_train.shape[1]
    mask = ~np.eye(n, dtype=bool)
    vals = np.concatenate([adj_dtf_train[i][mask] for i in range(len(adj_dtf_train))])
    return float(np.percentile(vals, percentile))


def normalize_adjacency(adj):
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = np.diag(d_inv_sqrt)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


def build_graphs(node_feats, adj_dtf, threshold):
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        ahat = normalize_adjacency(adj)
        x    = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, ahat))
    return graphs


def scale_node_features(graphs_train, graphs_val, graphs_test):
    all_x  = np.concatenate([g[0].numpy() for g in graphs_train], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_x)
    def apply(gs):
        return [(torch.tensor(scaler.transform(x.numpy()),
                              dtype=torch.float32), a)
                for x, a in gs]
    return apply(graphs_train), apply(graphs_val), apply(graphs_test)


# ─────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────

def aug_edge_dropout(x, ahat, p=0.20):
    mask = torch.rand_like(ahat) >= p
    diag = torch.arange(ahat.shape[0])
    mask[diag, diag] = True
    a_aug   = ahat * mask.float()
    row_sum = a_aug.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return x, a_aug / row_sum


def aug_node_noise(x, ahat, sigma=0.10):
    return x + torch.randn_like(x) * sigma, ahat


def aug_band_mask(x, ahat):
    """Zero out one random frequency band for all nodes (EEG-specific)."""
    band_idx = int(torch.randint(0, N_BAND_FEATS, (1,)).item())
    x_aug = x.clone()
    x_aug[:, band_idx] = 0.0
    return x_aug, ahat


ALL_AUGMENTATIONS = [aug_edge_dropout, aug_node_noise, aug_band_mask]


def random_augment(x, ahat):
    chosen = np.random.choice(len(ALL_AUGMENTATIONS), size=2, replace=False)
    x1, a1 = ALL_AUGMENTATIONS[chosen[0]](x, ahat)
    x2, a2 = ALL_AUGMENTATIONS[chosen[1]](x, ahat)
    return x1, a1, x2, a2


# ─────────────────────────────────────────────────────────────
# NT-Xent loss
# ─────────────────────────────────────────────────────────────

def nt_xent_loss(z1, z2, temperature=0.5):
    N  = z1.shape[0]
    z  = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([torch.arange(N, 2*N, device=z.device),
                        torch.arange(0, N,   device=z.device)])
    return F.cross_entropy(sim, labels)


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, x, ahat):
        return F.relu(self.W(ahat @ x))


class GCNEncoder(nn.Module):
    """2-layer GCN + global mean pool → graph embedding of size `hidden`."""
    def __init__(self, in_dim=N_NODE_FEATS, hidden=32, dropout=0.4):
        super().__init__()
        self.gcn1  = GCNLayer(in_dim, hidden)
        self.gcn2  = GCNLayer(hidden, hidden)
        self.drop  = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x, ahat):
        h = self.gcn1(x, ahat)
        h = self.drop(h)
        h = self.gcn2(h, ahat)
        return h.mean(dim=0)   # (hidden,) graph embedding


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=32, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, proj_dim)
        )
    def forward(self, h):
        return self.net(h)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=32, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(16, 1),
        )
    def forward(self, h):
        return self.net(h).squeeze()


# ─────────────────────────────────────────────────────────────
# SSL pre-training
# ─────────────────────────────────────────────────────────────

def ssl_pretrain(encoder, proj_head, train_graphs, ssl_epochs,
                 lr, device, batch_size=32, temperature=0.5):
    """Pre-train on TRAIN graphs only — test/val graphs never seen."""
    encoder.train()
    proj_head.train()
    params    = list(encoder.parameters()) + list(proj_head.parameters())
    optimiser = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=ssl_epochs, eta_min=lr * 0.1)

    N      = len(train_graphs)
    losses = []

    for ep in range(ssl_epochs):
        idx     = np.random.permutation(N)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = idx[start: start + batch_size]
            if len(batch_idx) < 2:
                continue
            z1_list, z2_list = [], []
            for i in batch_idx:
                x, a = train_graphs[i]
                x1, a1, x2, a2 = random_augment(x, a)
                h1 = encoder(x1.to(device), a1.to(device))
                h2 = encoder(x2.to(device), a2.to(device))
                z1_list.append(proj_head(h1))
                z2_list.append(proj_head(h2))

            z1   = torch.stack(z1_list)
            z2   = torch.stack(z2_list)
            loss = nt_xent_loss(z1, z2, temperature)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            ep_loss  += loss.item()
            n_batches += 1

        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        losses.append(avg)

        if (ep + 1) % 50 == 0:
            print(f'    SSL ep {ep+1:4d}/{ssl_epochs}  NT-Xent: {avg:.4f}')

    return losses


# ─────────────────────────────────────────────────────────────
# Fine-tuning (two phases)
# ─────────────────────────────────────────────────────────────

def finetune(encoder, clf_head, graphs_train, y_train,
             graphs_val, y_val, ft_epochs, lr, pos_weight,
             device, patience=20, freeze_encoder=False):
    """
    Phase A (freeze_encoder=True) : linear probe, warm up head only.
    Phase B (freeze_encoder=False): full fine-tune, all layers.
    Early stopping on VAL LOSS — not val AUC (avoids leakage).
    """
    for p in encoder.parameters():
        p.requires_grad = not freeze_encoder
    params    = list(clf_head.parameters()) if freeze_encoder \
                else list(encoder.parameters()) + list(clf_head.parameters())
    optimiser = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, verbose=False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))

    tr_losses, va_losses = [], []
    tr_aucs,   va_aucs   = [], []
    best_val_loss = np.inf
    best_enc_state = best_clf_state = None
    patience_cnt   = 0

    for ep in range(ft_epochs):
        encoder.train(); clf_head.train()
        ep_loss = 0.0
        for i in np.random.permutation(len(graphs_train)):
            x, a = graphs_train[i]
            x, a = x.to(device), a.to(device)
            optimiser.zero_grad()
            logit = clf_head(encoder(x, a))
            label = torch.tensor(float(y_train[i]), device=device).unsqueeze(0)
            loss  = criterion(logit.unsqueeze(0), label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            ep_loss += loss.item()
        tr_losses.append(ep_loss / len(graphs_train))

        # ── Eval ──────────────────────────────────────────────
        encoder.eval(); clf_head.eval()
        with torch.no_grad():
            tr_logits = np.array([
                clf_head(encoder(x.to(device), a.to(device))).cpu().item()
                for x, a in graphs_train
            ])
            va_logits = np.array([
                clf_head(encoder(x.to(device), a.to(device))).cpu().item()
                for x, a in graphs_val
            ])
        tr_probs = 1.0 / (1.0 + np.exp(-tr_logits))
        va_probs = 1.0 / (1.0 + np.exp(-va_logits))

        va_labels_t = torch.tensor(y_val, dtype=torch.float32)
        va_logits_t = torch.tensor(va_logits, dtype=torch.float32)
        va_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight))(va_logits_t, va_labels_t).item()
        va_losses.append(va_loss)

        tr_auc = float(roc_auc_score(y_train, tr_probs)) \
                 if len(np.unique(y_train)) > 1 else 0.0
        va_auc = float(roc_auc_score(y_val, va_probs)) \
                 if len(np.unique(y_val)) > 1 else 0.0
        tr_aucs.append(tr_auc)
        va_aucs.append(va_auc)

        scheduler.step(va_loss)

        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            best_enc_state = copy.deepcopy(encoder.state_dict())
            best_clf_state = copy.deepcopy(clf_head.state_dict())
            patience_cnt   = 0
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            print(f'    Early stop at epoch {ep+1} (val loss patience)')
            break

    if best_enc_state:
        encoder.load_state_dict(best_enc_state)
    if best_clf_state:
        clf_head.load_state_dict(best_clf_state)

    return tr_losses, va_losses, tr_aucs, va_aucs


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

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


@torch.no_grad()
def evaluate(encoder, clf_head, graphs, device):
    encoder.eval(); clf_head.eval()
    logits = np.array([
        clf_head(encoder(x.to(device), a.to(device))).cpu().item()
        for x, a in graphs
    ], dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.int64)
    return probs, preds


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_ssl_loss(ssl_losses, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ssl_losses, color='mediumpurple', lw=2)
    ax.set_xlabel('SSL Epoch', fontsize=11)
    ax.set_ylabel('NT-Xent Loss', fontsize=11)
    ax.set_title('SSL Pre-training Loss (Train split only)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ssl_loss_pretrain.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_ft_curves(tr_losses, va_losses, tr_aucs, va_aucs, phase, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(tr_losses, color='royalblue', lw=1.5, label='Train loss')
    axes[0].plot(va_losses, color='tomato',    lw=1.5, linestyle='--',
                 label='Val loss')
    gap = abs(tr_losses[-1] - va_losses[-1])
    axes[0].text(0.63, 0.88, f'Final gap: {gap:.4f}',
                 transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('BCE Loss', fontsize=11)
    axes[0].set_title(f'SSL-GCN Fine-tune ({phase}) — Loss',
                      fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tr_aucs, color='royalblue', lw=1.5, label='Train AUC')
    axes[1].plot(va_aucs, color='tomato',    lw=1.5, linestyle='--',
                 label='Val AUC')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('AUC', fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f'SSL-GCN Fine-tune ({phase}) — AUC',
                      fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'ft_curves_{phase.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, output_dir, cmap='Purples'):
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
        ax.set_title(f'SSL-GCN — Test CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'cm_ssl_test.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(val_metrics, test_metrics, fpr_va, tpr_va, fpr_te, tpr_te,
             output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_va, tpr_va, lw=2, color='steelblue',
            label=f'Val  AUC={val_metrics["auc"]:.3f}')
    ax.plot(fpr_te, tpr_te, lw=2, color='tomato',
            label=f'Test AUC={test_metrics["auc"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('SSL-GCN — ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_ssl.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting(train_auc, val_auc, test_auc, output_dir):
    """Train / Val / Test AUC bar chart with gap annotation."""
    labels = ['SSL-GCN']
    x, w   = np.arange(1), 0.25
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x - w,  [train_auc], w, label='Train AUC',
           color='mediumpurple', alpha=0.85, edgecolor='black')
    ax.bar(x,      [val_auc],   w, label='Val AUC',
           color='seagreen', alpha=0.85, edgecolor='black')
    ax.bar(x + w,  [test_auc],  w, label='Test AUC',
           color='tomato', alpha=0.85, edgecolor='black')
    gap   = train_auc - test_auc
    color = 'red' if gap > 0.10 else 'black'
    ax.text(0, max(train_auc, val_auc, test_auc) + 0.04,
            f'Tr-Te gap={gap:.3f}', ha='center',
            fontsize=11, fontweight='bold', color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('SSL-GCN — Overfitting: Train / Val / Test AUC',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_ssl.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  overfitting_ssl.png')


def plot_final_comparison(ssl_test_metrics, sup_json, baseline_json, output_dir):
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    models   = {}

    if baseline_json and Path(baseline_json).exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if res.get('test_metrics'):
                models[name] = res['test_metrics']

    if sup_json and Path(sup_json).exists():
        with open(sup_json) as f:
            sup = json.load(f)
        for name, res in sup.items():
            if res.get('test_metrics'):
                models[f'GCN ({name})'] = res['test_metrics']

    models['SSL-GCN (Ours)'] = ssl_test_metrics

    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    x      = np.arange(len(met_keys))
    width  = 0.15
    n      = len(models)

    fig, ax = plt.subplots(figsize=(13, 5))
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
    ax.set_title('Full Model Comparison — Test Set',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_all_models.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  comparison_all_models.png')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Step 6 — SSL GCN (TUH, fixed split)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--output_dir',    default='results/ssl_gnn')
    parser.add_argument('--ssl_epochs',    type=int,   default=200)
    parser.add_argument('--ft_epochs',     type=int,   default=100)
    parser.add_argument('--lr_ssl',        type=float, default=0.001)
    parser.add_argument('--lr_ft',         type=float, default=0.0005)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--proj_dim',      type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold_pct', type=float, default=70.0)
    parser.add_argument('--temperature',   type=float, default=0.5)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--baseline_json', default=None)
    parser.add_argument('--sup_json',      default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 6 — SSL GCN  (TUH, 22 ch, fixed 60-20-20 split)')
    print('=' * 65)
    print(f'Device : {device}')
    print(f'SSL epochs : {args.ssl_epochs}  LR : {args.lr_ssl}  '
          f'Temp : {args.temperature}  Batch : {args.batch_size}')
    print(f'FT  epochs : {args.ft_epochs}   LR : {args.lr_ft}  '
          f'Patience : {args.patience}')
    print(f'Threshold pct : {args.threshold_pct}  '
          f'(top {100-args.threshold_pct:.0f}% edges, TRAIN only)')
    print()
    print('LEAKAGE-FREE protocol:')
    print('  - Threshold  : computed from TRAIN adjacency only')
    print('  - Scaler     : fit on TRAIN node features only')
    print('  - SSL        : pre-trained on TRAIN graphs only (no labels)')
    print('  - Early stop : val LOSS (not val AUC)')

    # ── Load ──────────────────────────────────────────────────
    data       = np.load(args.featfile, allow_pickle=True)
    node_feats = data['node_features'].astype(np.float32)
    adj_dtf    = data['adj_dtf'].astype(np.float32)
    y          = data['y'].astype(np.int64)
    splits     = data['splits']

    tr = splits == 'train'
    va = splits == 'val'
    te = splits == 'test'

    print(f'\nTrain : {tr.sum():,}  Val : {va.sum():,}  Test : {te.sum():,}')
    majority_b = max((y[te] == 0).sum(), (y[te] == 1).sum()) / te.sum()
    print(f'Majority-class accuracy baseline (test): {majority_b*100:.1f}%')

    y_tr, y_va, y_te = y[tr], y[va], y[te]

    # ── Threshold (TRAIN only) ────────────────────────────────
    threshold = compute_threshold(adj_dtf[tr], args.threshold_pct)
    print(f'DTF threshold (p{args.threshold_pct:.0f} of train): {threshold:.4f}')

    # ── Build + scale graphs ──────────────────────────────────
    g_tr_raw = build_graphs(node_feats[tr], adj_dtf[tr], threshold)
    g_va_raw = build_graphs(node_feats[va], adj_dtf[va], threshold)
    g_te_raw = build_graphs(node_feats[te], adj_dtf[te], threshold)
    g_tr, g_va, g_te = scale_node_features(g_tr_raw, g_va_raw, g_te_raw)

    n_neg     = int((y_tr == 0).sum())
    n_pos     = int((y_tr == 1).sum())
    pos_weight= n_neg / (n_pos + 1e-12)
    print(f'Train: epilepsy={n_pos}, control={n_neg}, pos_weight={pos_weight:.2f}')

    # ── Phase 1: SSL pre-training (TRAIN only) ────────────────
    print(f'\n{"─"*55}')
    print(f'Phase 1 — SSL pre-training ({args.ssl_epochs} epochs, '
          f'{len(g_tr)} train graphs)')
    print(f'{"─"*55}')
    encoder  = GCNEncoder(in_dim=N_NODE_FEATS, hidden=args.hidden,
                          dropout=args.dropout).to(device)
    proj_head = ProjectionHead(in_dim=args.hidden,
                               proj_dim=args.proj_dim).to(device)
    ssl_losses = ssl_pretrain(
        encoder, proj_head, g_tr,
        ssl_epochs=args.ssl_epochs, lr=args.lr_ssl,
        device=device, batch_size=args.batch_size,
        temperature=args.temperature,
    )
    print(f'  SSL final NT-Xent loss: {ssl_losses[-1]:.4f}')
    plot_ssl_loss(ssl_losses, output_dir)

    # Discard projection head — keep encoder weights
    clf_head = ClassifierHead(in_dim=args.hidden,
                              dropout=args.dropout).to(device)

    # ── Phase 2A: Linear probe (frozen encoder) ───────────────
    print(f'\n{"─"*55}')
    print('Phase 2A — Linear probe (encoder frozen, 30 epochs)')
    print(f'{"─"*55}')
    tr_l, va_l, tr_a, va_a = finetune(
        encoder, clf_head, g_tr, y_tr, g_va, y_va,
        ft_epochs=30, lr=args.lr_ft * 5,
        pos_weight=pos_weight, device=device,
        patience=args.patience, freeze_encoder=True,
    )
    plot_ft_curves(tr_l, va_l, tr_a, va_a, 'PhaseA', output_dir)

    # ── Phase 2B: Full fine-tuning ────────────────────────────
    print(f'\n{"─"*55}')
    print(f'Phase 2B — Full fine-tuning ({args.ft_epochs} epochs)')
    print(f'{"─"*55}')
    tr_l2, va_l2, tr_a2, va_a2 = finetune(
        encoder, clf_head, g_tr, y_tr, g_va, y_va,
        ft_epochs=args.ft_epochs, lr=args.lr_ft,
        pos_weight=pos_weight, device=device,
        patience=args.patience, freeze_encoder=False,
    )
    plot_ft_curves(tr_l2, va_l2, tr_a2, va_a2, 'PhaseB', output_dir)

    # ── Final evaluation ──────────────────────────────────────
    tr_probs, tr_preds = evaluate(encoder, clf_head, g_tr, device)
    va_probs, va_preds = evaluate(encoder, clf_head, g_va, device)
    te_probs, te_preds = evaluate(encoder, clf_head, g_te, device)

    train_auc   = float(roc_auc_score(y_tr, tr_probs)) \
                  if len(np.unique(y_tr)) > 1 else 0.0
    val_metrics  = compute_metrics(y_va, va_preds, va_probs)
    test_metrics = compute_metrics(y_te, te_preds, te_probs)

    gap  = train_auc - test_metrics['auc']
    flag = '  ⚠ OVERFIT' if gap > 0.10 else ''

    print(f'\n  Train AUC : {train_auc:.3f}')
    print(f'  Val   AUC : {val_metrics["auc"]:.3f}')
    print(f'  Test  AUC : {test_metrics["auc"]:.3f}  '
          f'Gap={gap:.3f}{flag}')
    print(f'  Test  F1  : {test_metrics["f1"]:.3f}  '
          f'Sens={test_metrics["sensitivity"]:.3f}  '
          f'Spec={test_metrics["specificity"]:.3f}  '
          f'MCC={test_metrics["mcc"]:.3f}')

    # ── Plots ─────────────────────────────────────────────────
    cm = confusion_matrix(y_te, te_preds)
    plot_confusion_matrix(cm, output_dir)

    fpr_va, tpr_va, _ = roc_curve(y_va, va_probs)
    fpr_te, tpr_te, _ = roc_curve(y_te, te_probs)
    plot_roc(val_metrics, test_metrics, fpr_va, tpr_va, fpr_te, tpr_te, output_dir)
    plot_overfitting(train_auc, val_metrics['auc'], test_metrics['auc'], output_dir)
    plot_final_comparison(test_metrics, args.sup_json, args.baseline_json, output_dir)

    # ── Save ──────────────────────────────────────────────────
    results = dict(
        model              = 'SSL-GCN (GraphCL, TUH)',
        hyperparameters    = vars(args),
        ssl_protocol       = 'Pre-trained on TRAIN graphs only — NO leakage',
        threshold_note     = f'p{args.threshold_pct:.0f} percentile of TRAIN DTF values',
        augmentations      = ['edge_dropout(p=0.20)', 'node_noise(σ=0.10)',
                               'band_mask(random_band) — EEG-specific, our contribution'],
        early_stopping     = 'val loss (not val AUC — avoids leakage)',
        train_auc          = train_auc,
        val_metrics        = val_metrics,
        test_metrics       = test_metrics,
        overfit_gap        = round(gap, 4),
        ssl_final_loss     = float(ssl_losses[-1]),
    )
    results_path = output_dir / 'results_ssl.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  ✓ Saved: {results_path}')

    print('\n' + '=' * 65)
    print('STEP 6 COMPLETE')
    print('=' * 65)


if __name__ == '__main__':
    main()