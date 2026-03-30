"""
Step 6 — Self-Supervised GCN  (TUH Dataset, GraphCL, Fixed Split)
==================================================================
DATASET CONTEXT:
  TUH epilepsy detection — binary classification: epilepsy (1) vs control (0)
  200 patients, ~193,000 epochs, 22 channels, 250 Hz, 17 node features
  Evaluation: fixed 60-20-20 patient-stratified split (from step 0)

KEY DIFFERENCES FROM TUC PIPELINE:
  - Evaluation: fixed split instead of 8-fold LOPO
  - SSL pre-training uses ALL training epochs (no LOPO loop needed)
  - 22 nodes, 17 node features, 1000 samples per epoch
  - Much larger dataset → SSL has more unlabelled graphs to learn from
  - Task: epilepsy vs control (not ictal vs pre-ictal)

WHY SSL ON TUH:
  Even with 200 patients, labels require clinical expert annotation and
  may not perfectly capture the full diversity of epileptic EEG patterns.
  SSL pre-training on all training graphs (without labels) allows the
  encoder to learn the general structure of EEG connectivity graphs
  before being told which patients have epilepsy. This is the same
  motivation as TUC but with a stronger case — the larger unlabelled
  pool (116k training epochs) should give the contrastive objective a
  much richer negative sampling distribution than TUC's ~900 epochs.
  A batch of 64 gives 63 negatives per anchor — more than TUC's 31.

DATA LEAKAGE PREVENTION:
  - SSL pre-training uses ONLY training epochs (val/test never seen)
  - Graph threshold computed from training adjacency only
  - Feature scaler fitted on training set only
  - Early stopping on val LOSS (not val AUC — avoids checkpoint leakage)
  - Val patients never seen during SSL pre-training

OVERFITTING PREVENTION — SIX METHODS:
  1. Early stopping on val loss (patience configurable)
  2. Dropout in encoder + classifier head (p=0.4)
  3. Weight decay L2 λ=1e-4 (Adam)
  4. Gradient clipping max_norm=1.0
  5. ReduceLROnPlateau for fine-tuning LR
  6. Two-phase fine-tuning: linear probe first (30 epochs, encoder frozen)
     then full fine-tune (encoder unfrozen, lower LR)
     The linear probe prevents the randomly-initialised classifier head
     from immediately destroying the pre-trained encoder representations.

SSL STRATEGY: GraphCL (You et al., 2020)
  Positive pairs  : two augmented views of the same graph
  Negative pairs  : all other graphs in the batch
  Loss            : NT-Xent (Chen et al., 2020)
  Batch size      : 64 (doubles TUC's 32 — more negatives per anchor)

AUGMENTATIONS (three, EEG-motivated):
  1. Edge dropout (p=0.20)   — intermittent functional connectivity
  2. Node feature noise (σ=0.10) — electrode noise / amplitude variation
  3. Band masking (random band zeroed) — seizure band varies across patients

ARCHITECTURE:
  GCNEncoder: GCNLayer(17→32) → Dropout → GCNLayer(32→32) → MeanPool
  ProjectionHead: Linear(32→32) → ReLU → Linear(32→proj_dim)
  ClassifierHead: Linear(32→16) → ReLU → Dropout → Linear(16→1)

  Identical encoder to SmallGCN in step 5 — ensures fair comparison.

Outputs:
  results_ssl.json              all metrics (train/val/test)
  ssl_pretrain_loss.png         NT-Xent loss curve over SSL epochs
  loss_curve_ssl.png            fine-tuning loss + AUC curves
  roc_ssl_gcn.png               ROC curves (val + test)
  cm_ssl_{partition}.png        confusion matrices
  overfitting_ssl.png           train/val/test AUC bar chart
  per_patient_ssl.png           per-patient test AUC
  comparison_all_models.png     RF + SVM + SmallGCN + PureGCN + SSL-GCN

Usage:
  python step6_ssl_gnn.py \\
      --featfile      F:\\features\\features_all.npz \\
      --outputdir     F:\\results\\ssl_gnn \\
      --baseline_json F:\\results\\baseline_ml\\results_baseline.json \\
      --sup_json      F:\\results\\gnn_supervised\\results_gcn.json \\
      --ssl_epochs    200 \\
      --ft_epochs     100 \\
      --lr_ssl        0.001 \\
      --lr_ft         0.0005 \\
      --hidden        32 \\
      --dropout       0.4 \\
      --threshold_pct 70 \\
      --temperature   0.5 \\
      --batch_size    64 \\
      --patience      20
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
NODE_DIM    = 17
N_BAND_FEATS = 7     # TUH has 7 frequency bands (vs 6 in TUC)


# ══════════════════════════════════════════════════════════════
# 2. ADJACENCY UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_threshold(adj_dtf_train: np.ndarray,
                      percentile: float = 70.0) -> float:
    """
    Data-driven edge threshold from training adjacency only.
    Never touches val or test patients.
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


def build_graphs(node_feats: np.ndarray,
                 adj_dtf: np.ndarray,
                 threshold: float) -> list:
    """Build graph list: (x_tensor, a_hat_tensor) per epoch."""
    graphs = []
    for i in range(len(node_feats)):
        adj = adj_dtf[i].copy()
        adj[adj < threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        a_hat = normalize_adjacency(adj)
        x     = torch.tensor(node_feats[i], dtype=torch.float32)
        graphs.append((x, a_hat))
    return graphs


def scale_node_features(graphs_train: list,
                         graphs_val:   list,
                         graphs_test:  list) -> tuple:
    """Fit StandardScaler on train only, apply to all three splits."""
    all_train_x = np.concatenate([g[0].numpy() for g in graphs_train], axis=0)
    scaler      = StandardScaler()
    scaler.fit(all_train_x)

    def apply(graphs):
        return [
            (torch.tensor(scaler.transform(x.numpy()), dtype=torch.float32), a)
            for x, a in graphs
        ]
    return apply(graphs_train), apply(graphs_val), apply(graphs_test)


# ══════════════════════════════════════════════════════════════
# 3. EEG-SPECIFIC AUGMENTATIONS
# ══════════════════════════════════════════════════════════════

def augment_edge_dropout(x: torch.Tensor, a: torch.Tensor,
                          p: float = 0.20):
    """
    Randomly zero edges with probability p.
    Self-loops preserved for GCN stability.
    Reflects intermittent functional coupling between brain regions.
    """
    mask                       = (torch.rand_like(a) > p).float()
    diag                       = torch.arange(a.shape[0])
    mask[diag, diag]           = 1.0
    a_aug                      = a * mask
    row_sum                    = a_aug.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return x, a_aug / row_sum


def augment_node_noise(x: torch.Tensor, a: torch.Tensor,
                        sigma: float = 0.10):
    """
    Gaussian noise on node features.
    Mimics electrode noise and minor amplitude fluctuations.
    """
    return x + torch.randn_like(x) * sigma, a


def augment_band_mask(x: torch.Tensor, a: torch.Tensor):
    """
    Zero one randomly selected frequency band for ALL nodes.
    Band features are indices 0..N_BAND_FEATS-1 of node feature vector.
    EEG-specific: forces learning band-invariant representations.
    The seizure-related spectral signature varies across patients
    and this augmentation prevents the encoder from relying on one band.
    """
    band_idx         = int(torch.randint(0, N_BAND_FEATS, (1,)).item())
    x_aug            = x.clone()
    x_aug[:, band_idx] = 0.0
    return x_aug, a


ALL_AUGMENTATIONS = [augment_edge_dropout, augment_node_noise, augment_band_mask]


def random_augment(x: torch.Tensor, a: torch.Tensor):
    """
    Apply two DISTINCT augmentations independently to produce positive pair.
    Each view uses one augmentation applied to the original graph.
    """
    chosen   = np.random.choice(len(ALL_AUGMENTATIONS), size=2, replace=False)
    x1, a1  = ALL_AUGMENTATIONS[chosen[0]](x.clone(), a.clone())
    x2, a2  = ALL_AUGMENTATIONS[chosen[1]](x.clone(), a.clone())
    return (x1, a1), (x2, a2)


# ══════════════════════════════════════════════════════════════
# 4. NT-XENT CONTRASTIVE LOSS
# ══════════════════════════════════════════════════════════════

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                  temperature: float = 0.5) -> torch.Tensor:
    """
    Normalised Temperature-Scaled Cross-Entropy loss.
    (Chen et al., 2020 — SimCLR)

    z1, z2   : (N, proj_dim) embeddings of two augmented views.
    Positive : (i, i+N) and (i+N, i)
    Negative : all other 2N-2 pairs within the batch.

    With batch_size=64: 63 negatives per anchor.
    More negatives than TUC (31) → stronger contrastive signal.
    Temperature τ=0.5 is the standard default.
    """
    N   = z1.shape[0]
    z   = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2N, proj_dim)
    sim = torch.mm(z, z.T) / temperature                   # (2N, 2N)
    # Mask self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ══════════════════════════════════════════════════════════════
# 5. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(a @ x))


class GCNEncoder(nn.Module):
    """
    2-layer GCN → GlobalMeanPool → graph embedding (hidden,).

    Identical to SmallGCN encoder in step 5 for fair comparison.
    In SSL pre-training: feeds into ProjectionHead.
    In fine-tuning: feeds into ClassifierHead.
    """
    def __init__(self, in_dim: int = NODE_DIM,
                 hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.gcn1    = GCNLayer(in_dim, hidden)
        self.gcn2    = GCNLayer(hidden, hidden)
        self.drop    = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, a)
        h = self.drop(h)
        h = self.gcn2(h, a)
        return h.mean(dim=0)   # (hidden,) — global mean pool


class ProjectionHead(nn.Module):
    """
    MLP projection head used ONLY during SSL pre-training.
    Discarded after pre-training — only encoder weights are kept.
    Keeping it separate ensures the encoder representations are not
    distorted by the contrastive objective after fine-tuning begins.
    """
    def __init__(self, in_dim: int = 32, proj_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ClassifierHead(nn.Module):
    """Binary classifier attached to encoder during fine-tuning."""
    def __init__(self, in_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze()


# ══════════════════════════════════════════════════════════════
# 6. SSL PRE-TRAINING
# ══════════════════════════════════════════════════════════════

def ssl_pretrain(encoder: GCNEncoder,
                  proj_head: ProjectionHead,
                  train_graphs: list,
                  ssl_epochs: int,
                  lr: float,
                  device: torch.device,
                  batch_size: int = 64,
                  temperature: float = 0.5) -> list:
    """
    Pre-train encoder + projection head with NT-Xent loss.

    LEAKAGE-FREE: train_graphs contains ONLY training-split graphs.
    Val and test patients' graphs are never passed here.

    Uses CosineAnnealingLR: LR decays smoothly from lr to lr*0.1
    over ssl_epochs. This is more appropriate than ReduceLROnPlateau
    for contrastive learning where loss oscillates more.

    Returns list of per-epoch average NT-Xent losses.
    """
    encoder.train()
    proj_head.train()

    params    = list(encoder.parameters()) + list(proj_head.parameters())
    optimiser = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=ssl_epochs,
                                  eta_min=lr * 0.1)

    N      = len(train_graphs)
    losses = []

    print(f'  SSL pre-training: {ssl_epochs} epochs, '
          f'batch={batch_size}, τ={temperature}')
    print(f'  Training graphs: {N}  '
          f'(~{N // batch_size} batches/epoch, '
          f'{batch_size - 1} negatives per anchor)')

    for ep in range(1, ssl_epochs + 1):
        idx       = np.random.permutation(N)
        ep_loss   = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            batch_idx = idx[start: start + batch_size]
            if len(batch_idx) < 2:
                continue

            z1_list, z2_list = [], []
            for i in batch_idx:
                x, a               = train_graphs[i]
                (x1, a1), (x2, a2) = random_augment(x, a)
                h1 = encoder(x1.to(device), a1.to(device))
                h2 = encoder(x2.to(device), a2.to(device))
                z1_list.append(proj_head(h1))
                z2_list.append(proj_head(h2))

            z1   = torch.stack(z1_list)
            z2   = torch.stack(z2_list)
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()

            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        losses.append(avg)

        if ep % 25 == 0 or ep == 1:
            print(f'    SSL [{ep:4d}/{ssl_epochs}]  '
                  f'NT-Xent: {avg:.4f}  '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')

    return losses


# ══════════════════════════════════════════════════════════════
# 7. FINE-TUNING  (two phases)
# ══════════════════════════════════════════════════════════════

def finetune_phase(encoder: GCNEncoder,
                    clf_head: ClassifierHead,
                    graphs_train: list, y_train: np.ndarray,
                    graphs_val:   list, y_val:   np.ndarray,
                    ft_epochs: int,
                    lr: float,
                    pos_weight: float,
                    device: torch.device,
                    patience: int = 20,
                    freeze_encoder: bool = False,
                    phase_name: str = '') -> tuple:
    """
    One fine-tuning phase.

    Phase A (freeze_encoder=True) — linear probe:
      Only the classifier head trains. Encoder is frozen.
      Warms up the classifier so Phase B starts from a reasonable
      initialisation, preventing the random head from destroying
      the pre-trained encoder representations immediately.

    Phase B (freeze_encoder=False) — full fine-tuning:
      All parameters train with weight decay and lower LR.
      ReduceLROnPlateau applied to val loss.

    OVERFITTING PREVENTION in both phases:
      - Early stopping on val LOSS (not val AUC)
      - Weight decay L2 λ=1e-4
      - Gradient clipping max_norm=1.0
      - ReduceLROnPlateau for fine-tuning LR

    Returns: train_losses, val_losses, train_aucs, val_aucs, best_val_loss
    """
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        params = list(clf_head.parameters())
    else:
        for p in encoder.parameters():
            p.requires_grad = True
        params = list(encoder.parameters()) + list(clf_head.parameters())

    optimiser  = Adam(params, lr=lr, weight_decay=1e-4)
    scheduler  = ReduceLROnPlateau(optimiser, patience=10, factor=0.5,
                                   verbose=False)
    criterion  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    train_losses, val_losses = [], []
    train_aucs,   val_aucs   = [], []
    best_val_loss  = np.inf
    best_enc_state = None
    best_clf_state = None
    patience_cnt   = 0

    for ep in range(1, ft_epochs + 1):
        # ── Train ────────────────────────────────────────────────
        encoder.train()
        clf_head.train()
        ep_loss = 0.0
        for i in np.random.permutation(len(graphs_train)):
            x, a  = graphs_train[i]
            optimiser.zero_grad()
            logit = clf_head(encoder(x.to(device), a.to(device)))
            label = torch.tensor(float(y_train[i]),
                                 device=device).unsqueeze(0)
            loss  = criterion(logit.unsqueeze(0), label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            ep_loss += loss.item()
        train_losses.append(ep_loss / max(len(graphs_train), 1))

        # ── Evaluate ─────────────────────────────────────────────
        encoder.eval()
        clf_head.eval()

        with torch.no_grad():
            tr_logits = np.array([
                clf_head(encoder(x.to(device), a.to(device))).cpu().item()
                for x, a in graphs_train
            ])
            val_logits = np.array([
                clf_head(encoder(x.to(device), a.to(device))).cpu().item()
                for x, a in graphs_val
            ])

        tr_probs  = 1.0 / (1.0 + np.exp(-tr_logits))
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        tr_auc  = float(roc_auc_score(y_train, tr_probs)) \
                  if len(np.unique(y_train)) == 2 else 0.0
        val_auc = float(roc_auc_score(y_val, val_probs)) \
                  if len(np.unique(y_val)) == 2 else 0.0
        train_aucs.append(tr_auc)
        val_aucs.append(val_auc)

        # Val loss for early stopping
        val_labels_t  = torch.tensor(y_val, dtype=torch.float32)
        val_logits_t  = torch.tensor(val_logits, dtype=torch.float32)
        val_loss      = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )(val_logits_t, val_labels_t).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Early stopping on val LOSS
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_enc_state = copy.deepcopy(encoder.state_dict())
            best_clf_state = copy.deepcopy(clf_head.state_dict())
            patience_cnt   = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'    {phase_name} early stop at epoch {ep}')
                break

    # Restore best checkpoint
    if best_enc_state:
        encoder.load_state_dict(best_enc_state)
    if best_clf_state:
        clf_head.load_state_dict(best_clf_state)

    return train_losses, val_losses, train_aucs, val_aucs, best_val_loss


# ══════════════════════════════════════════════════════════════
# 8. EVALUATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(encoder: GCNEncoder, clf_head: ClassifierHead,
             graphs: list, labels: np.ndarray,
             device: torch.device) -> tuple:
    encoder.eval()
    clf_head.eval()
    logits = np.array([
        clf_head(encoder(x.to(device), a.to(device))).cpu().item()
        for x, a in graphs
    ], dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    return probs, preds


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
# 9. PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def plot_ssl_loss(ssl_losses: list, ssl_epochs: int,
                  output_dir: Path):
    """
    NT-Xent loss curve over SSL pre-training.
    A healthy curve decreases from ~log(2N-1) toward 0.
    If the curve barely moves, the contrastive signal is too weak
    (batch too small, augmentations too strong, or LR too low).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(ssl_losses) + 1), ssl_losses,
            color='mediumpurple', lw=2)
    ax.axhline(np.log(2 * 64 - 1), color='gray', linestyle='--',
               lw=1.5, label='Chance (log(2N−1) for N=64)')
    ax.set_xlabel('SSL Epoch', fontsize=12)
    ax.set_ylabel('NT-Xent Loss', fontsize=12)
    ax.set_title('SSL Pre-training — NT-Xent Loss\n'
                 '(should decrease from chance level)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate start and end
    ax.annotate(f'Start: {ssl_losses[0]:.3f}',
                xy=(1, ssl_losses[0]),
                xytext=(ssl_epochs * 0.1, ssl_losses[0] + 0.05),
                fontsize=9, color='gray')
    ax.annotate(f'End: {ssl_losses[-1]:.3f}',
                xy=(len(ssl_losses), ssl_losses[-1]),
                xytext=(ssl_epochs * 0.6, ssl_losses[-1] + 0.05),
                fontsize=9, color='mediumpurple')

    plt.tight_layout()
    plt.savefig(output_dir / 'ssl_pretrain_loss.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ ssl_pretrain_loss.png')


def plot_finetune_curves(train_losses_a, val_losses_a,
                          train_aucs_a,  val_aucs_a,
                          train_losses_b, val_losses_b,
                          train_aucs_b,  val_aucs_b,
                          output_dir: Path):
    """
    Four-panel fine-tuning diagnostic:
      Left column : loss curves (Phase A and B concatenated)
      Right column: AUC curves
    Phase boundary marked with a vertical line.
    """
    n_a = len(train_losses_a)
    n_b = len(train_losses_b)

    tr_loss = train_losses_a + train_losses_b
    vl_loss = val_losses_a   + val_losses_b
    tr_auc  = train_aucs_a   + train_aucs_b
    vl_auc  = val_aucs_a     + val_aucs_b
    epochs  = range(1, len(tr_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Loss
    axes[0].plot(epochs, tr_loss, color='steelblue', lw=1.5,
                 label='Train loss')
    axes[0].plot(epochs, vl_loss, color='tomato', lw=1.5,
                 linestyle='--', label='Val loss')
    if n_a > 0:
        axes[0].axvline(n_a, color='gray', linestyle=':',
                        lw=1.5, label='Phase A→B')
    axes[0].set_xlabel('Fine-tune epoch', fontsize=11)
    axes[0].set_ylabel('BCE Loss', fontsize=11)
    axes[0].set_title('SSL-GCN — Fine-tune Loss', fontsize=11,
                      fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # AUC
    axes[1].plot(epochs, tr_auc, color='steelblue', lw=1.5,
                 label='Train AUC')
    axes[1].plot(epochs, vl_auc, color='tomato', lw=1.5,
                 linestyle='--', label='Val AUC')
    if n_a > 0:
        axes[1].axvline(n_a, color='gray', linestyle=':',
                        lw=1.5, label='Phase A→B')
    axes[1].set_ylim(0.4, 1.05)
    axes[1].set_xlabel('Fine-tune epoch', fontsize=11)
    axes[1].set_ylabel('AUC', fontsize=11)
    axes[1].set_title('SSL-GCN — Fine-tune AUC', fontsize=11,
                      fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Annotate final gap
    if vl_loss:
        gap = abs(tr_loss[-1] - vl_loss[-1])
        axes[0].text(0.63, 0.88, f'Final loss gap: {gap:.3f}',
                     transform=axes[0].transAxes, fontsize=9,
                     bbox=dict(boxstyle='round',
                               facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve_ssl.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ loss_curve_ssl.png')


def plot_roc_curves(y_val, y_prob_val,
                    y_test, y_prob_test,
                    output_dir: Path):
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
    ax.set_title('SSL-GCN — ROC Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_ssl_gcn.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ roc_ssl_gcn.png')


def plot_confusion_matrix(y_true, y_pred, partition: str,
                           output_dir: Path):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Purples', ax=ax,
                    xticklabels=['Control', 'Epilepsy'],
                    yticklabels=['Control', 'Epilepsy'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'SSL-GCN — {partition} CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_ssl_{partition.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting(results: dict, output_dir: Path):
    """Train / Val / Test AUC bar chart for SSL-GCN."""
    partitions = ['train', 'val', 'test']
    colors     = {'train': 'steelblue',
                  'val':   'darkorange',
                  'test':  'tomato'}
    x     = np.arange(1)
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, p in enumerate(partitions):
        auc = results.get(p, {}).get('auc', 0.0)
        ax.bar(x + (i - 1) * width, [auc], width,
               label=p.capitalize(),
               color=colors[p], alpha=0.85, edgecolor='black')

    tr = results.get('train', {}).get('auc', 0)
    te = results.get('test',  {}).get('auc', 0)
    gap = tr - te
    col = 'red' if gap > 0.10 else 'black'
    ax.text(0, max(tr, te) + 0.05,
            f'Train-Test Δ={gap:.2f}',
            ha='center', fontsize=11, fontweight='bold', color=col)

    ax.set_xticks([0])
    ax.set_xticklabels(['SSL-GCN'], fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('SSL-GCN — Train / Val / Test AUC',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_ssl.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ overfitting_ssl.png')


def plot_per_patient_auc(y_true: np.ndarray, y_prob: np.ndarray,
                          patient_ids: np.ndarray,
                          output_dir: Path):
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
    colors   = ['tomato' if a < 0.5 else 'mediumpurple'
                for a in pat_aucs[sort_idx]]

    fig, ax = plt.subplots(figsize=(max(10, len(pat_aucs) * 0.4), 5))
    ax.barh(range(len(pat_aucs)), pat_aucs[sort_idx],
            color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5, label='Chance')
    ax.axvline(float(np.mean(pat_aucs)), color='navy', linestyle='-',
               lw=2, label=f'Mean = {np.mean(pat_aucs):.3f}')
    ax.set_yticks(range(len(pat_aucs)))
    ax.set_yticklabels(np.array(patients)[sort_idx], fontsize=7)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'SSL-GCN — Per-Patient Test AUC\n'
                 f'(red = below chance; {(pat_aucs < 0.5).sum()} patients)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_patient_ssl.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ per_patient_ssl.png')


def plot_all_models_comparison(ssl_test_metrics: dict,
                                baseline_json: Path,
                                sup_json: Path,
                                output_dir: Path):
    """
    Final comparison: RF + SVM + SmallGCN + PureGCN + SSL-GCN.
    Test-set metrics only.
    """
    all_stats = {}

    if baseline_json and baseline_json.exists():
        with open(baseline_json) as f:
            bl = json.load(f)
        for name, res in bl.items():
            if res.get('test'):
                all_stats[name] = res['test']

    if sup_json and sup_json.exists():
        with open(sup_json) as f:
            sup = json.load(f)
        if sup.get('test'):
            all_stats['SmallGCN'] = sup['test']

    all_stats['SSL-GCN'] = ssl_test_metrics

    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    colors   = ['steelblue', 'tomato', 'seagreen',
                'mediumpurple', 'darkorange']
    x        = np.arange(len(met_keys))
    n        = len(all_stats)
    bar_w    = 0.15

    fig, ax = plt.subplots(figsize=(13, 5))
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


def plot_ssl_vs_supervised(ssl_results: dict,
                            sup_json: Path,
                            output_dir: Path):
    """
    Direct comparison: SmallGCN (supervised) vs SSL-GCN.
    Shows all three partitions to reveal whether SSL helped
    generalisation or just shifted the train/val/test balance.
    """
    if not (sup_json and sup_json.exists()):
        return

    with open(sup_json) as f:
        sup = json.load(f)

    models = {
        'SmallGCN\n(supervised)': sup,
        'SSL-GCN':                ssl_results,
    }

    partitions = ['train', 'val', 'test']
    colors     = {'train': 'steelblue',
                  'val':   'darkorange',
                  'test':  'tomato'}
    x          = np.arange(len(models))
    width      = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, partition in enumerate(partitions):
        aucs = [m.get(partition, {}).get('auc', 0.0)
                for m in models.values()]
        ax.bar(x + (i - 1) * width, aucs, width,
               label=partition.capitalize(),
               color=colors[partition], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(list(models.keys()), fontsize=11)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('Supervised GCN vs SSL-GCN\n'
                 'Train / Val / Test AUC',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'ssl_vs_supervised.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ ssl_vs_supervised.png')


# ══════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 6 — SSL GCN (TUH dataset, GraphCL)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',      required=True)
    parser.add_argument('--outputdir',     default='results/ssl_gnn')
    parser.add_argument('--baseline_json', default=None)
    parser.add_argument('--sup_json',      default=None,
                        help='Path to results_gcn.json from step 5')
    parser.add_argument('--ssl_epochs',    type=int,   default=200)
    parser.add_argument('--ft_epochs',     type=int,   default=100)
    parser.add_argument('--lr_ssl',        type=float, default=0.001)
    parser.add_argument('--lr_ft',         type=float, default=0.0005)
    parser.add_argument('--hidden',        type=int,   default=32)
    parser.add_argument('--proj_dim',      type=int,   default=32)
    parser.add_argument('--dropout',       type=float, default=0.4)
    parser.add_argument('--threshold_pct', type=float, default=70.0)
    parser.add_argument('--temperature',   type=float, default=0.5)
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--patience',      type=int,   default=20)
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 65)
    print('STEP 6 — SSL GCN  (TUH dataset, GraphCL)')
    print('=' * 65)
    print(f'Device        : {device}')
    print(f'SSL epochs    : {args.ssl_epochs}  LR: {args.lr_ssl}  '
          f'τ: {args.temperature}')
    print(f'FT epochs     : {args.ft_epochs}   LR: {args.lr_ft}')
    print(f'Hidden        : {args.hidden}  ProjDim: {args.proj_dim}  '
          f'Dropout: {args.dropout}')
    print(f'Batch size    : {args.batch_size}  '
          f'(negatives per anchor: {args.batch_size - 1})')
    print(f'Threshold pct : {args.threshold_pct}')
    print(f'\nOVERFITTING PREVENTION:')
    print(f'  1. Early stopping on val LOSS (patience={args.patience})')
    print(f'  2. Dropout p={args.dropout} (encoder + head)')
    print(f'  3. Weight decay L2 λ=1e-4')
    print(f'  4. Gradient clipping max_norm=1.0')
    print(f'  5. ReduceLROnPlateau in fine-tuning')
    print(f'  6. Two-phase fine-tune (linear probe → full)')
    print(f'\nLEAKAGE-FREE PROTOCOL:')
    print(f'  SSL pre-training on training graphs ONLY')
    print(f'  Threshold from training adjacency only')
    print(f'  Scaler fit on training set only')
    print(f'  Early stopping on val LOSS (not val AUC)')
    print('=' * 65)

    # ── Load features ──────────────────────────────────────────────────
    data        = np.load(args.featfile, allow_pickle=True)
    node_feats  = data['node_features'].astype(np.float32)  # (N, 22, 17)
    adj_dtf     = data['adj_dtf'].astype(np.float32)        # (N, 22, 22)
    y           = data['y'].astype(np.int64)
    splits      = data['splits']
    patient_ids = data['patient_ids']

    train_mask = splits == 'train'
    val_mask   = splits == 'val'
    test_mask  = splits == 'test'

    y_train = y[train_mask]
    y_val   = y[val_mask]
    y_test  = y[test_mask]
    pat_test = patient_ids[test_mask]

    majority_b = max((y == 0).sum(), (y == 1).sum()) / len(y)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    pos_weight = n_neg / (n_pos + 1e-12)

    print(f'\nDataset: {len(y):,} epochs total')
    print(f'  Train : {train_mask.sum():,}  Val: {val_mask.sum():,}  '
          f'Test: {test_mask.sum():,}')
    print(f'  Train balance — epi: {n_pos}, ctrl: {n_neg}  '
          f'(pos_weight={pos_weight:.3f})')
    print(f'  Majority-class baseline: {majority_b * 100:.1f}%\n')

    # ── Threshold from training adjacency only ─────────────────────────
    print('Computing DTF threshold from training adjacency...')
    threshold = compute_threshold(adj_dtf[train_mask], args.threshold_pct)
    print(f'  Threshold (p{args.threshold_pct:.0f}): {threshold:.4f}\n')

    # ── Build graphs ───────────────────────────────────────────────────
    print('Building graphs...')
    g_train_raw = build_graphs(
        node_feats[train_mask], adj_dtf[train_mask], threshold)
    g_val_raw   = build_graphs(
        node_feats[val_mask],   adj_dtf[val_mask],   threshold)
    g_test_raw  = build_graphs(
        node_feats[test_mask],  adj_dtf[test_mask],  threshold)

    g_train, g_val, g_test = scale_node_features(
        g_train_raw, g_val_raw, g_test_raw)
    print(f'  Graphs — train: {len(g_train)}, '
          f'val: {len(g_val)}, test: {len(g_test)}')

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — SSL PRE-TRAINING (training graphs only)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print('PHASE 1 — SSL PRE-TRAINING')
    print('─' * 60)

    encoder   = GCNEncoder(in_dim=NODE_DIM, hidden=args.hidden,
                           dropout=args.dropout).to(device)
    proj_head = ProjectionHead(in_dim=args.hidden,
                               proj_dim=args.proj_dim).to(device)

    ssl_losses = ssl_pretrain(
        encoder, proj_head,
        g_train,
        ssl_epochs  = args.ssl_epochs,
        lr          = args.lr_ssl,
        device      = device,
        batch_size  = args.batch_size,
        temperature = args.temperature,
    )

    plot_ssl_loss(ssl_losses, args.ssl_epochs, output_dir)
    print(f'\n  SSL complete. Loss: {ssl_losses[0]:.4f} → {ssl_losses[-1]:.4f}')

    # Projection head discarded — only encoder weights kept
    proj_head = None

    # ══════════════════════════════════════════════════════════════
    # PHASE 2A — LINEAR PROBE (encoder frozen, 30 epochs)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print('PHASE 2A — LINEAR PROBE (encoder frozen)')
    print('─' * 60)

    clf_head = ClassifierHead(in_dim=args.hidden,
                               dropout=args.dropout).to(device)

    tr_loss_a, vl_loss_a, tr_auc_a, vl_auc_a, _ = finetune_phase(
        encoder, clf_head,
        g_train, y_train,
        g_val,   y_val,
        ft_epochs      = 30,
        lr             = args.lr_ft * 5,   # higher LR for head only
        pos_weight     = pos_weight,
        device         = device,
        patience       = args.patience,
        freeze_encoder = True,
        phase_name     = 'Linear probe',
    )
    print(f'  Linear probe done: {len(tr_loss_a)} epochs  '
          f'best val AUC={max(vl_auc_a):.3f}')

    # ══════════════════════════════════════════════════════════════
    # PHASE 2B — FULL FINE-TUNING (all layers, lower LR)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print('PHASE 2B — FULL FINE-TUNING')
    print('─' * 60)

    tr_loss_b, vl_loss_b, tr_auc_b, vl_auc_b, best_val = finetune_phase(
        encoder, clf_head,
        g_train, y_train,
        g_val,   y_val,
        ft_epochs      = args.ft_epochs,
        lr             = args.lr_ft,
        pos_weight     = pos_weight,
        device         = device,
        patience       = args.patience,
        freeze_encoder = False,
        phase_name     = 'Full fine-tune',
    )
    print(f'  Full fine-tune done: {len(tr_loss_b)} epochs  '
          f'best val loss={best_val:.4f}')

    plot_finetune_curves(
        tr_loss_a, vl_loss_a, tr_auc_a, vl_auc_a,
        tr_loss_b, vl_loss_b, tr_auc_b, vl_auc_b,
        output_dir,
    )

    # ── Final evaluation on all three partitions ───────────────────────
    print('\n' + '─' * 60)
    print('FINAL EVALUATION')
    print('─' * 60)

    all_results = {}
    all_probs   = {}

    for partition, graphs, labels in [
        ('train', g_train, y_train),
        ('val',   g_val,   y_val),
        ('test',  g_test,  y_test),
    ]:
        probs, preds = evaluate(encoder, clf_head, graphs, labels, device)
        m = compute_metrics(labels, preds, probs, partition)
        all_results[partition] = m
        all_probs[partition]   = (probs, preds, labels)

        print(f'  {partition:5s} | AUC={m.get("auc",0):.3f}  '
              f'F1={m.get("f1",0):.3f}  '
              f'Sens={m.get("sensitivity",0):.3f}  '
              f'Spec={m.get("specificity",0):.3f}  '
              f'Acc={m.get("accuracy",0):.3f}')

    tr_auc = all_results['train'].get('auc', 0)
    te_auc = all_results['test'].get('auc', 0)
    gap    = tr_auc - te_auc
    flag   = ' ⚠ OVERFIT' if gap > 0.10 else ' ✓ OK'
    print(f'\n  Train-Test AUC gap: {gap:.3f}{flag}')

    # ── Plots ──────────────────────────────────────────────────────────
    val_probs_arr,  val_preds_arr,  _ = all_probs['val']
    test_probs_arr, test_preds_arr, _ = all_probs['test']

    plot_roc_curves(y_val, val_probs_arr, y_test, test_probs_arr, output_dir)

    for partition, (probs, preds, targets) in all_probs.items():
        plot_confusion_matrix(targets, preds, partition, output_dir)

    plot_overfitting(all_results, output_dir)
    plot_per_patient_auc(y_test, test_probs_arr, pat_test, output_dir)

    bl_path  = Path(args.baseline_json) if args.baseline_json else None
    sup_path = Path(args.sup_json)      if args.sup_json      else None
    plot_all_models_comparison(all_results['test'], bl_path, sup_path,
                               output_dir)
    plot_ssl_vs_supervised(all_results, sup_path, output_dir)

    # ── Save results ───────────────────────────────────────────────────
    results_out = {
        'model':            'SSL_GCN_GraphCL',
        'hyperparameters':  vars(args),
        'ssl_protocol':     'pre-training on training graphs only (no leakage)',
        'augmentations':    [
            'edge_dropout(p=0.20)',
            'node_noise(sigma=0.10)',
            'band_mask(random, TUH-specific 7 bands)',
        ],
        'ssl_loss_start':   float(ssl_losses[0]),
        'ssl_loss_end':     float(ssl_losses[-1]),
        'train_test_gap':   float(gap),
        **all_results,
    }

    results_path = output_dir / 'results_ssl.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f'\n  ✓ Results → {results_path}')

    # ── Final summary ──────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('STEP 6 COMPLETE — FINAL SUMMARY')
    print('=' * 65)
    print(f'{"Partition":8s} | {"AUC":>7} {"F1":>7} '
          f'{"Sens":>7} {"Spec":>7} {"Acc":>7}')
    print('-' * 50)
    for p in ['train', 'val', 'test']:
        m = all_results.get(p, {})
        print(f'{p:8s} | {m.get("auc",0):7.3f} {m.get("f1",0):7.3f} '
              f'{m.get("sensitivity",0):7.3f} '
              f'{m.get("specificity",0):7.3f} '
              f'{m.get("accuracy",0):7.3f}')
    print(f'\nTrain-Test AUC gap : {gap:.3f}{flag}')
    print(f'SSL NT-Xent        : {ssl_losses[0]:.4f} → {ssl_losses[-1]:.4f}')
    print(f'Majority-class acc : {majority_b * 100:.1f}%')
    print('=' * 65)


if __name__ == '__main__':
    main()
