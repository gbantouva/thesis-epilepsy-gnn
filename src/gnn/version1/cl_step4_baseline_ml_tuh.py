"""
Step 4 — Baseline ML: Random Forest + SVM  (TUH Dataset, Fixed Split)
======================================================================
DATASET CONTEXT:
  TUH epilepsy detection — binary classification: epilepsy (1) vs control (0)
  200 patients, ~193,000 epochs, 58 hand-crafted features
  Evaluation: fixed 60-20-20 patient-stratified split (from step 0)
  No LOPO — dataset is too large; fixed split is standard for TUH scale.

KEY DIFFERENCES FROM TUC PIPELINE:
  - Evaluation: fixed train/val/test split instead of LOPO
  - Features: 58 flat features (vs 53 in TUC)
  - Task: epilepsy vs control (vs ictal vs pre-ictal in TUC)
  - Scale: ~193k epochs across 200 patients (vs ~1069 in TUC)
  - No per-fold threshold computation (no graph construction here)

OVERFITTING PREVENTION STRATEGY:
  This script uses FOUR complementary methods to prevent and detect
  overfitting, which is the primary risk when models see many epochs
  from few patients:

  1. CONSERVATIVE HYPERPARAMETERS
     RF : max_depth=8, min_samples_leaf=10, max_features='sqrt'
          → Shallow trees, large leaf requirement, random feature subsets.
          These constrain individual tree complexity relative to dataset size.
     SVM: C=0.1, RBF kernel, gamma='scale'
          → Low C = strong regularisation, wider margin, less sensitivity
          to individual training points. C=0.1 (not 1.0) is more aggressive
          regularisation appropriate for the scale of this dataset.

  2. PATIENT-STRATIFIED SPLIT (from step 0)
     No patient appears in more than one partition. All epochs from the
     same patient go to the same split. This is the critical leakage fix —
     without it, a model trained on patient X's other recordings would
     trivially generalise to patient X's test recording.

  3. FEATURE SCALING FITTED ON TRAIN ONLY
     StandardScaler is fit on training epochs and applied to val/test.
     This is standard practice but explicitly enforced here.

  4. OVERFITTING ANALYSIS PLOTS
     For every model: train vs val vs test AUC comparison, feature
     importance (RF), learning curve (performance vs training size),
     confusion matrices, ROC curves. The gap between train and val AUC
     is the primary overfitting diagnostic.

EVALUATION PROTOCOL:
  Primary evaluation is on the TEST set (unseen patients).
  Val set is used for overfitting diagnostics only — NOT for tuning.
  Metrics: AUC, F1, Sensitivity, Specificity, Accuracy, MCC, Precision.
  Accuracy is reported alongside the majority-class baseline.

Outputs (saved to --outputdir):
  results_baseline.json              all metrics, train/val/test
  summary_table.csv                  mean metrics per model
  roc_{model}.png                    ROC curves (val + test)
  cm_{model}_{partition}.png         confusion matrices
  feature_importance_{model}.png     RF feature importances
  learning_curve_{model}.png         performance vs training size
  overfitting_summary.png            train/val/test AUC comparison
  per_patient_test_{model}.png       per-patient test AUC (test patients)
  calibration_{model}.png            probability calibration plot

Usage:
  python step4_baseline_ml.py \\
      --featfile  F:\\features\\features_all.npz \\
      --outputdir F:\\results\\baseline_ml
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.base as skbase

from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.svm import SVC

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# 1. METRIC HELPERS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    partition: str = '') -> dict:
    """
    Full classification metric dict.

    Accuracy is always reported alongside majority-class baseline so
    it is never mistaken for a meaningful standalone metric on an
    imbalanced dataset.
    """
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
# 2. PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def plot_roc_curves(y_val, y_prob_val,
                    y_test, y_prob_test,
                    model_name: str, output_dir: Path):
    """
    ROC curves for val and test partitions on the same axes.
    A large gap between val and test ROC curves indicates that the
    model generalises differently to val and test patients —
    this signals instability rather than overfitting per se.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for y_true, y_prob, label, color, ls in [
        (y_val,  y_prob_val,  'Validation', 'steelblue',  '-'),
        (y_test, y_prob_test, 'Test',       'tomato',     '--'),
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


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str, partition: str,
                           output_dir: Path):
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


def plot_feature_importance(importances: np.ndarray,
                             feature_names: list,
                             model_name: str, output_dir: Path,
                             top_n: int = 30):
    """
    Bar chart of top-N RF feature importances.
    Shows which feature groups (spectral / hjorth / graph) are most
    discriminative for epilepsy vs control classification.
    """
    idx      = np.argsort(importances)[::-1][:top_n]
    top_imp  = importances[idx]
    top_names = np.array(feature_names)[idx]

    # Color bars by feature group
    colors = []
    for name in top_names:
        if name.startswith('spec_'):
            colors.append('steelblue')
        elif name.startswith('hjorth_'):
            colors.append('tomato')
        else:
            colors.append('seagreen')

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(top_n), top_imp, color=colors,
                  edgecolor='black', alpha=0.85)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_names, rotation=90, fontsize=7)
    ax.set_ylabel('Feature importance (Gini)', fontsize=11)
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances\n'
                 f'Blue=Spectral  Red=Hjorth  Green=Graph',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    tag = model_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'feature_importance_{tag}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curve(model, model_name: str,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray,   y_val: np.ndarray,
                         output_dir: Path,
                         n_steps: int = 8):
    """
    Performance vs training size — the most informative overfitting plot.

    If val AUC plateaus or drops while train AUC keeps rising as more
    training data is added, the model is memorising rather than learning.
    If both curves rise together, the model benefits from more data.

    Uses fractions of the training set: 10%, 20%, ..., 100%.
    Each point trains a fresh model clone — no state between points.
    """
    fractions = np.linspace(0.10, 1.0, n_steps)
    train_aucs, val_aucs = [], []

    n_train = len(X_train)
    rng     = np.random.RandomState(42)

    for frac in fractions:
        n_sub   = max(int(n_train * frac), 50)
        sub_idx = rng.choice(n_train, n_sub, replace=False)

        clf = skbase.clone(model)
        clf.fit(X_train[sub_idx], y_train[sub_idx])

        if hasattr(clf, 'predict_proba'):
            tr_prob  = clf.predict_proba(X_train[sub_idx])[:, 1]
            val_prob = clf.predict_proba(X_val)[:, 1]
        else:
            raw_tr  = clf.decision_function(X_train[sub_idx])
            raw_val = clf.decision_function(X_val)
            tr_prob  = (raw_tr  - raw_tr.min())  / (raw_tr.max()  - raw_tr.min()  + 1e-12)
            val_prob = (raw_val - raw_val.min()) / (raw_val.max() - raw_val.min() + 1e-12)

        train_aucs.append(float(roc_auc_score(y_train[sub_idx], tr_prob)))
        val_aucs.append(float(roc_auc_score(y_val, val_prob)))

    n_sizes = [int(n_train * f) for f in fractions]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_sizes, train_aucs, 'o-', color='steelblue', lw=2,
            label='Train AUC')
    ax.plot(n_sizes, val_aucs,   's--', color='tomato',   lw=2,
            label='Val AUC')
    ax.fill_between(n_sizes, train_aucs, val_aucs, alpha=0.10, color='gray',
                    label='Overfit gap')
    ax.axhline(0.5, color='gray', linestyle=':', lw=1, label='Chance')
    ax.set_xlabel('Training epochs', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f'{model_name} — Learning Curve\n'
                 f'(convergence = curves meet; overfit = gap widens)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    tag = model_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'learning_curve_{tag}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ learning_curve_{tag}.png')


def plot_overfitting_summary(results: dict, output_dir: Path):
    """
    Three-partition AUC bar chart for all models side by side.

    This is the primary overfitting diagnostic figure.
    A model that generalises well shows similar bars across all three
    partitions. A large train-test gap signals overfitting.
    A large val-test gap signals instability in the split.
    """
    met_keys   = ['auc', 'f1', 'sensitivity', 'specificity']
    model_names = list(results.keys())
    partitions  = ['train', 'val', 'test']
    colors      = {'train': 'steelblue', 'val': 'darkorange', 'test': 'tomato'}

    # ── AUC comparison bar chart ──────────────────────────────────────
    x     = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, partition in enumerate(partitions):
        aucs = [results[m][partition].get('auc', 0.0) for m in model_names]
        ax.bar(x + (i - 1) * width, aucs, width,
               label=partition.capitalize(),
               color=colors[partition], alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('Train / Val / Test AUC — All Models\n'
                 '(small gap = good generalisation)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate gaps
    for i, m in enumerate(model_names):
        tr  = results[m]['train'].get('auc', 0)
        te  = results[m]['test'].get('auc', 0)
        gap = tr - te
        col = 'red' if gap > 0.10 else 'black'
        ax.text(i, max(tr, te) + 0.04, f'Δ={gap:.2f}',
                ha='center', fontsize=10, fontweight='bold', color=col)

    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ overfitting_summary.png')

    # ── Full metric comparison (test set only) ────────────────────────
    colors_met = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    x2 = np.arange(len(met_keys))
    n  = len(model_names)
    bar_w = 0.20

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(model_names):
        means  = [results[m]['test'].get(k, 0.0) for k in met_keys]
        offset = (i - n / 2 + 0.5) * bar_w
        ax.bar(x2 + offset, means, bar_w,
               label=m, color=colors_met[i % len(colors_met)],
               alpha=0.85, edgecolor='black')

    ax.set_xticks(x2)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score (test set)', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('Test Set Performance — All Models',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_test.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ model_comparison_test.png')


def plot_per_patient_auc(y_true: np.ndarray, y_prob: np.ndarray,
                          patient_ids: np.ndarray,
                          model_name: str, output_dir: Path):
    """
    Per-patient AUC on the test set.

    This is critical for a large multi-patient dataset. A high mean AUC
    can hide systematic failure on specific patient subgroups (e.g. a
    patient whose EEG looks unlike the training distribution).
    Patients with AUC < 0.5 are highlighted in red — these are harder
    than random chance and deserve attention in the Discussion.
    """
    patients    = np.unique(patient_ids)
    pat_aucs    = []
    pat_labels  = []

    for pat in patients:
        mask = patient_ids == pat
        if len(np.unique(y_true[mask])) < 2:
            continue   # only one class — can't compute AUC
        try:
            auc = float(roc_auc_score(y_true[mask], y_prob[mask]))
            pat_aucs.append(auc)
            pat_labels.append(pat)
        except Exception:
            continue

    if len(pat_aucs) == 0:
        return

    pat_aucs   = np.array(pat_aucs)
    sort_idx   = np.argsort(pat_aucs)
    sorted_auc = pat_aucs[sort_idx]
    sorted_lbl = np.array(pat_labels)[sort_idx]

    colors = ['tomato' if a < 0.5 else 'steelblue' for a in sorted_auc]

    fig, ax = plt.subplots(figsize=(max(10, len(pat_aucs) * 0.4), 5))
    ax.barh(range(len(sorted_auc)), sorted_auc, color=colors,
            edgecolor='black', alpha=0.85)
    ax.axvline(0.5,  color='gray', linestyle='--', lw=1.5, label='Chance')
    ax.axvline(float(np.mean(pat_aucs)), color='navy', linestyle='-',
               lw=2, label=f'Mean AUC = {np.mean(pat_aucs):.3f}')
    ax.set_yticks(range(len(sorted_auc)))
    ax.set_yticklabels(sorted_lbl, fontsize=7)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'{model_name} — Per-Patient Test AUC\n'
                 f'(red = below chance; {(pat_aucs < 0.5).sum()} patients)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    tag = model_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'per_patient_test_{tag}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ per_patient_test_{tag}.png')


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                      model_name: str, partition: str,
                      output_dir: Path):
    """
    Reliability / calibration plot.

    A well-calibrated model has predicted probabilities that match
    actual class frequencies. The diagonal is perfect calibration.
    Points above the diagonal = model is underconfident.
    Points below = overconfident (common in SVMs and RFs).
    Overconfident models are more likely to have poor probability
    thresholding in clinical use.
    """
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform'
        )
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    ax.plot(mean_pred, fraction_pos, 'o-', color='steelblue',
            lw=2, ms=7, label=model_name)
    ax.set_xlabel('Mean predicted probability', fontsize=12)
    ax.set_ylabel('Fraction of positives (epilepsy)', fontsize=12)
    ax.set_title(f'{model_name} — Calibration Curve ({partition})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    tag = model_name.lower().replace(' ', '_')
    plt.savefig(output_dir / f'calibration_{tag}_{partition.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════
# 3. PROBABILITY EXTRACTION
# ══════════════════════════════════════════════════════════════

def get_probs(clf, X: np.ndarray) -> np.ndarray:
    """
    Extract calibrated [0, 1] probabilities from any sklearn classifier.
    SVM with probability=True uses Platt scaling — still well-defined.
    """
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    raw = clf.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


# ══════════════════════════════════════════════════════════════
# 4. MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════

def evaluate_model(model_name: str,
                   model,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val:   np.ndarray, y_val:   np.ndarray,
                   X_test:  np.ndarray, y_test:  np.ndarray,
                   patient_ids_test: np.ndarray,
                   feature_names: list,
                   output_dir: Path) -> dict:
    """
    Train once on train split, evaluate on train / val / test.
    Generates all overfitting plots for this model.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each containing a metric dict,
    plus 'importances' for RF.
    """
    print(f'\n{"=" * 60}')
    print(f'  {model_name}')
    print(f'{"=" * 60}')

    # ── Fit ────────────────────────────────────────────────────────────
    print('  Fitting on training set...')
    model.fit(X_train, y_train)
    print(f'  Done.')

    # ── Predictions ────────────────────────────────────────────────────
    prob_train = get_probs(model, X_train)
    prob_val   = get_probs(model, X_val)
    prob_test  = get_probs(model, X_test)

    pred_train = (prob_train >= 0.5).astype(int)
    pred_val   = (prob_val   >= 0.5).astype(int)
    pred_test  = (prob_test  >= 0.5).astype(int)

    # ── Metrics ────────────────────────────────────────────────────────
    m_train = compute_metrics(y_train, pred_train, prob_train, 'train')
    m_val   = compute_metrics(y_val,   pred_val,   prob_val,   'val')
    m_test  = compute_metrics(y_test,  pred_test,  prob_test,  'test')

    train_auc = m_train.get('auc', 0)
    val_auc   = m_val.get('auc', 0)
    test_auc  = m_test.get('auc', 0)
    gap_tv    = train_auc - val_auc
    gap_tt    = train_auc - test_auc

    print(f'\n  {"Partition":8s} | {"AUC":>6} {"F1":>6} '
          f'{"Sens":>6} {"Spec":>6} {"Acc":>6} {"MCC":>6}')
    print(f'  {"-"*55}')
    for name, m in [('Train', m_train), ('Val', m_val), ('Test', m_test)]:
        print(f'  {name:8s} | {m.get("auc",0):6.3f} {m.get("f1",0):6.3f} '
              f'{m.get("sensitivity",0):6.3f} {m.get("specificity",0):6.3f} '
              f'{m.get("accuracy",0):6.3f} {m.get("mcc",0):6.3f}')

    gap_flag = ' ⚠ OVERFIT' if gap_tt > 0.10 else ''
    print(f'\n  Train-Val  gap : {gap_tv:.3f}')
    print(f'  Train-Test gap : {gap_tt:.3f}{gap_flag}')

    # ── Plots ──────────────────────────────────────────────────────────
    tag = model_name.lower().replace(' ', '_')

    plot_roc_curves(y_val, prob_val, y_test, prob_test,
                    model_name, output_dir)

    for y_p, y_pr, part in [
        (y_train, pred_train, 'train'),
        (y_val,   pred_val,   'val'),
        (y_test,  pred_test,  'test'),
    ]:
        plot_confusion_matrix(y_p, y_pr, model_name, part, output_dir)
        plot_calibration(y_p, get_probs(model, {
            'train': X_train, 'val': X_val, 'test': X_test
        }[part]), model_name, part, output_dir)

    plot_per_patient_auc(y_test, prob_test, patient_ids_test,
                         model_name, output_dir)

    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plot_feature_importance(importances, feature_names,
                                model_name, output_dir)

    return {
        'train':       m_train,
        'val':         m_val,
        'test':        m_test,
        'importances': importances.tolist() if importances is not None else None,
        'train_val_gap':  float(gap_tv),
        'train_test_gap': float(gap_tt),
    }


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 4 — Baseline ML (TUH dataset, fixed split)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',  required=True,
                        help='features/features_all.npz from step 3')
    parser.add_argument('--outputdir', default='results/baseline_ml',
                        help='Output directory for results and plots')
    parser.add_argument('--no_learning_curve', action='store_true',
                        help='Skip learning curve (slow for large datasets)')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('STEP 4 — BASELINE ML  (TUH, epilepsy vs control)')
    print('=' * 65)

    # ── Load features ──────────────────────────────────────────────────
    data          = np.load(args.featfile, allow_pickle=True)
    X             = data['X'].astype(np.float32)
    y             = data['y'].astype(np.int64)
    splits        = data['splits']
    patient_ids   = data['patient_ids']
    feature_names = data['feature_names'].tolist()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Split masks ────────────────────────────────────────────────────
    train_mask = splits == 'train'
    val_mask   = splits == 'val'
    test_mask  = splits == 'test'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    pat_train = patient_ids[train_mask]
    pat_val   = patient_ids[val_mask]
    pat_test  = patient_ids[test_mask]

    n_epi_train  = int((y_train == 1).sum())
    n_ctrl_train = int((y_train == 0).sum())
    majority_b   = max((y == 0).sum(), (y == 1).sum()) / len(y)

    print(f'\nDataset loaded:')
    print(f'  Features  : {X.shape[1]}')
    print(f'  Total     : {len(y):,} epochs')
    print(f'  Train     : {train_mask.sum():,} epochs  '
          f'({len(np.unique(pat_train))} patients,  '
          f'epi={n_epi_train}, ctrl={n_ctrl_train})')
    print(f'  Val       : {val_mask.sum():,} epochs  '
          f'({len(np.unique(pat_val))} patients)')
    print(f'  Test      : {test_mask.sum():,} epochs  '
          f'({len(np.unique(pat_test))} patients)')
    print(f'\nMajority-class baseline : {majority_b * 100:.1f}%')
    print(f'  (all-control predictor achieves this accuracy for free)')

    # ── Feature scaling ────────────────────────────────────────────────
    # Scaler fitted on train only — never sees val or test data
    print('\nFitting StandardScaler on training set...')
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print('  ✓ Scaler fitted and applied')

    # ── Model definitions ──────────────────────────────────────────────
    # OVERFITTING PREVENTION: conservative hyperparameters
    #
    # Random Forest:
    #   max_depth=8           — shallow trees reduce memorisation
    #   min_samples_leaf=10   — each leaf needs ≥10 samples (not 1)
    #   max_features='sqrt'   — random feature subsets per split
    #   n_estimators=300      — enough trees for stable importances
    #   class_weight='balanced'— compensates for epilepsy/control imbalance
    #
    # SVM RBF:
    #   C=0.1                 — stronger regularisation than default C=1
    #                           Large margin, less sensitivity to outliers
    #   gamma='scale'         — automatic scaling: 1/(n_features * X.var())
    #   class_weight='balanced'— compensates for class imbalance
    #   probability=True      — enables Platt-scaled probabilities for AUC

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'SVM RBF': SVC(
            kernel='rbf',
            C=0.1,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
        ),
    }

    print('\nOVERFITTING PREVENTION:')
    print('  RF : max_depth=8, min_samples_leaf=10, max_features=sqrt')
    print('  SVM: C=0.1 (aggressive regularisation), gamma=scale')
    print('  Both: class_weight=balanced, scaler fit on train only')
    print('  Split: strict patient separation (no patient in 2 partitions)')

    # ── Run evaluation ─────────────────────────────────────────────────
    all_results = {}

    for model_name, model in models.items():
        res = evaluate_model(
            model_name    = model_name,
            model         = model,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            patient_ids_test = pat_test,
            feature_names    = feature_names,
            output_dir       = output_dir,
        )
        all_results[model_name] = res

        # Learning curve (slow — skip with --no_learning_curve)
        if not args.no_learning_curve:
            print(f'\n  Computing learning curve for {model_name}...')
            print(f'  (use --no_learning_curve to skip if slow)')
            plot_learning_curve(
                model      = skbase.clone(models[model_name]),
                model_name = model_name,
                X_train    = X_train,
                y_train    = y_train,
                X_val      = X_val,
                y_val      = y_val,
                output_dir = output_dir,
            )

    # ── Summary plots ──────────────────────────────────────────────────
    plot_overfitting_summary(all_results, output_dir)

    # ── Save results JSON ──────────────────────────────────────────────
    results_path = output_dir / 'results_baseline.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  ✓ Results → {results_path}')

    # ── Summary table ──────────────────────────────────────────────────
    display_keys = ['auc', 'f1', 'sensitivity', 'specificity',
                    'accuracy', 'mcc']
    rows = []
    for model_name, res in all_results.items():
        for partition in ['train', 'val', 'test']:
            m = res[partition]
            if not m:
                continue
            row = {'Model': model_name, 'Partition': partition}
            for k in display_keys:
                row[k] = f"{m.get(k, 0):.3f}"
            rows.append(row)

    df = pd.DataFrame(rows)
    print('\n' + '=' * 65)
    print('FINAL RESULTS TABLE')
    print('=' * 65)
    print(df.to_string(index=False))
    df.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f'\n  ✓ Summary table → {output_dir / "summary_table.csv"}')

    # ── Overfitting diagnosis ──────────────────────────────────────────
    print('\n' + '=' * 65)
    print('OVERFITTING DIAGNOSIS')
    print('=' * 65)
    print(f'{"Model":20s} | {"Train AUC":>10} {"Val AUC":>9} '
          f'{"Test AUC":>9} | {"Gap (Tr-Te)":>12} {"Status":>12}')
    print('-' * 80)
    for model_name, res in all_results.items():
        tr   = res['train'].get('auc', 0)
        val  = res['val'].get('auc', 0)
        te   = res['test'].get('auc', 0)
        gap  = tr - te
        flag = '⚠ Overfit' if gap > 0.10 else '✓ OK'
        print(f'{model_name:20s} | {tr:10.3f} {val:9.3f} '
              f'{te:9.3f} | {gap:12.3f} {flag:>12}')

    print(f'\nMajority-class accuracy baseline : {majority_b * 100:.1f}%')
    print(f'A model predicting all-control achieves this for free.')
    print(f'Sensitivity is the most clinically important metric —')
    print(f'failing to detect epilepsy is worse than a false alarm.')

    print('\n' + '=' * 65)
    print('STEP 4 COMPLETE')
    print('=' * 65)
    print('\nNext:')
    print(f'  python step5_gnn_supervised.py --featfile {args.featfile}')


if __name__ == '__main__':
    main()
