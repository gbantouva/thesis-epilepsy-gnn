"""
Step 4b — SVM-RFE only (TUH Dataset, Fixed Split)
==================================================
Standalone script that runs SVM with Recursive Feature Elimination
on the TUH dataset using the same fixed split and feature file as
step4_baseline_ml_tuh.py.

Run this AFTER step4 — it does NOT re-run Random Forest or SVM-RBF.
Results are saved to a separate JSON so they can be merged with
the existing results_baseline.json if needed.

SVM-RFE: a linear SVM is used to rank all 58 features by their
contribution to classification. The weakest features are removed
iteratively until only the top N remain. A final RBF-SVM classifier
is then trained on only those N features. This tests whether
reducing the feature space improves cross-patient generalisation.

Hyperparameters:
  RFE estimator : LinearSVC, C=0.01 (strong regularisation)
  Features kept : top 15 out of 58
  Final SVM     : RBF kernel, C=0.01, gamma=scale, class_weight=balanced

Outputs (saved to --outputdir):
  results_svm_rfe.json               all metrics train/val/test
  roc_svm_rfe.png                    ROC curves (val + test)
  cm_svm_rfe_{partition}.png         confusion matrices
  feature_stability_svm_rfe.png      selected features bar chart
  learning_curve_svm_rfe.png         performance vs training size
  per_patient_test_svm_rfe.png       per-patient test AUC
  calibration_svm_rfe_{part}.png     calibration plots
  overfitting_svm_rfe.png            train/val/test AUC summary

Usage:
  python step4b_svm_rfe_tuh.py \\
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
from sklearn.feature_selection import RFE
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
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings('ignore')

MODEL_NAME = 'SVM RFE'
N_FEATURES_TO_SELECT = 15   # top features to keep out of 58


# ══════════════════════════════════════════════════════════════
# 1. METRIC HELPERS
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob, partition=''):
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


def get_probs(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    raw = clf.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


# ══════════════════════════════════════════════════════════════
# 2. PLOT HELPERS
# ══════════════════════════════════════════════════════════════

def plot_roc(y_val, y_prob_val, y_test, y_prob_test, output_dir):
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
    ax.set_title(f'{MODEL_NAME} — ROC Curves', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_svm_rfe.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ roc_svm_rfe.png')


def plot_confusion_matrix(y_true, y_pred, partition, output_dir):
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
        ax.set_title(f'{MODEL_NAME} — {partition} CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cm_svm_rfe_{partition.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ cm_svm_rfe_{partition.lower()}.png')


def plot_feature_stability(selected_mask, feature_names, output_dir):
    """Bar chart showing which features were selected by RFE."""
    selected = np.array(feature_names)[selected_mask]
    colors = []
    for name in selected:
        if name.startswith('spec_'):
            colors.append('steelblue')
        elif name.startswith('hjorth_'):
            colors.append('tomato')
        else:
            colors.append('seagreen')

    fig, ax = plt.subplots(figsize=(max(10, len(selected) * 0.6), 5))
    ax.bar(range(len(selected)), np.ones(len(selected)),
           color=colors, edgecolor='black', alpha=0.85)
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=90, fontsize=8)
    ax.set_ylabel('Selected (1 = kept)', fontsize=11)
    ax.set_title(f'{MODEL_NAME} — Selected Features '
                 f'({len(selected)}/{len(feature_names)})\n'
                 f'Blue=Spectral  Red=Hjorth  Green=Graph',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.4)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_stability_svm_rfe.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ feature_stability_svm_rfe.png')


def plot_learning_curve(rfe_selector, final_clf, X_train, y_train,
                        X_val, y_val, output_dir, n_steps=8):
    """Performance vs training size using the already-selected features."""
    X_tr_sel  = rfe_selector.transform(X_train)
    X_val_sel = rfe_selector.transform(X_val)

    fractions  = np.linspace(0.10, 1.0, n_steps)
    train_aucs, val_aucs = [], []
    n_train = len(X_tr_sel)
    rng = np.random.RandomState(42)

    for frac in fractions:
        n_sub   = max(int(n_train * frac), 50)
        sub_idx = rng.choice(n_train, n_sub, replace=False)
        clf = skbase.clone(final_clf)
        clf.fit(X_tr_sel[sub_idx], y_train[sub_idx])
        tr_prob  = get_probs(clf, X_tr_sel[sub_idx])
        val_prob = get_probs(clf, X_val_sel)
        train_aucs.append(float(roc_auc_score(y_train[sub_idx], tr_prob)))
        val_aucs.append(float(roc_auc_score(y_val, val_prob)))

    n_sizes = [int(n_train * f) for f in fractions]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_sizes, train_aucs, 'o-', color='steelblue', lw=2, label='Train AUC')
    ax.plot(n_sizes, val_aucs,   's--', color='tomato',   lw=2, label='Val AUC')
    ax.fill_between(n_sizes, train_aucs, val_aucs, alpha=0.10, color='gray',
                    label='Overfit gap')
    ax.axhline(0.5, color='gray', linestyle=':', lw=1, label='Chance')
    ax.set_xlabel('Training epochs', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_title(f'{MODEL_NAME} — Learning Curve', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curve_svm_rfe.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ learning_curve_svm_rfe.png')


def plot_per_patient_auc(y_true, y_prob, patient_ids, output_dir):
    patients = np.unique(patient_ids)
    pat_aucs, pat_labels = [], []
    for pat in patients:
        mask = patient_ids == pat
        if len(np.unique(y_true[mask])) < 2:
            continue
        try:
            auc = float(roc_auc_score(y_true[mask], y_prob[mask]))
            pat_aucs.append(auc)
            pat_labels.append(pat)
        except Exception:
            continue
    if not pat_aucs:
        return
    pat_aucs   = np.array(pat_aucs)
    sort_idx   = np.argsort(pat_aucs)
    sorted_auc = pat_aucs[sort_idx]
    sorted_lbl = np.array(pat_labels)[sort_idx]
    colors = ['tomato' if a < 0.5 else 'steelblue' for a in sorted_auc]
    fig, ax = plt.subplots(figsize=(max(10, len(pat_aucs) * 0.4), 5))
    ax.barh(range(len(sorted_auc)), sorted_auc, color=colors,
            edgecolor='black', alpha=0.85)
    ax.axvline(0.5, color='gray', linestyle='--', lw=1.5, label='Chance')
    ax.axvline(float(np.mean(pat_aucs)), color='navy', linestyle='-',
               lw=2, label=f'Mean AUC = {np.mean(pat_aucs):.3f}')
    ax.set_yticks(range(len(sorted_auc)))
    ax.set_yticklabels(sorted_lbl, fontsize=7)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title(f'{MODEL_NAME} — Per-Patient Test AUC\n'
                 f'(red = below chance; {(pat_aucs < 0.5).sum()} patients)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_patient_test_svm_rfe.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ per_patient_test_svm_rfe.png')


def plot_calibration(y_true, y_prob, partition, output_dir):
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy='uniform')
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    ax.plot(mean_pred, fraction_pos, 'o-', color='steelblue',
            lw=2, ms=7, label=MODEL_NAME)
    ax.set_xlabel('Mean predicted probability', fontsize=12)
    ax.set_ylabel('Fraction of positives (epilepsy)', fontsize=12)
    ax.set_title(f'{MODEL_NAME} — Calibration ({partition})',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'calibration_svm_rfe_{partition.lower()}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting_summary(m_train, m_val, m_test, output_dir):
    partitions = ['Train', 'Val', 'Test']
    aucs = [m_train['auc'], m_val['auc'], m_test['auc']]
    colors = ['steelblue', 'darkorange', 'tomato']
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(partitions, aucs, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.01,
                f'{auc:.3f}', ha='center', fontsize=12, fontweight='bold')
    gap = m_train['auc'] - m_test['auc']
    color = 'red' if gap > 0.10 else 'black'
    ax.set_title(f'{MODEL_NAME} — Train / Val / Test AUC\n'
                 f'Overfit gap (train-test): {gap:+.3f}',
                 fontsize=11, fontweight='bold', color=color)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_svm_rfe.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ overfitting_svm_rfe.png')


# ══════════════════════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 4b — SVM-RFE only (TUH dataset)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',  required=True,
                        help='features/features_all.npz from step 3')
    parser.add_argument('--outputdir', default='results/baseline_ml',
                        help='Output directory (same as step 4 recommended)')
    parser.add_argument('--n_features', type=int, default=N_FEATURES_TO_SELECT,
                        help=f'Number of features to keep (default: {N_FEATURES_TO_SELECT})')
    parser.add_argument('--no_learning_curve', action='store_true',
                        help='Skip learning curve (slow)')
    args = parser.parse_args()

    output_dir = Path(args.outputdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('STEP 4b — SVM-RFE  (TUH, epilepsy vs control)')
    print('=' * 65)

    # ── Load features ──────────────────────────────────────────────────
    print('\nLoading features...')
    data          = np.load(args.featfile, allow_pickle=True)
    X             = data['X'].astype(np.float32)
    y             = data['y'].astype(np.int64)
    splits        = data['splits']
    patient_ids   = data['patient_ids']
    feature_names = data['feature_names'].tolist()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    train_mask = splits == 'train'
    val_mask   = splits == 'val'
    test_mask  = splits == 'test'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    pat_test         = patient_ids[test_mask]

    majority_b = max((y == 0).sum(), (y == 1).sum()) / len(y)

    print(f'  Train : {train_mask.sum():,} epochs')
    print(f'  Val   : {val_mask.sum():,} epochs')
    print(f'  Test  : {test_mask.sum():,} epochs')
    print(f'  Features: {X.shape[1]}')
    print(f'  Majority baseline: {majority_b * 100:.1f}%')

    # ── Feature scaling — fit on train only ────────────────────────────
    print('\nFitting StandardScaler on training set...')
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print('  ✓ Scaler fitted and applied')

    # ── RFE: rank features using LinearSVC ────────────────────────────
    print(f'\nRunning RFE — selecting top {args.n_features} of '
          f'{X_train.shape[1]} features...')
    print('  (This may take a few minutes on 52k epochs)')

    rfe_estimator = LinearSVC(
        C=0.01,
        class_weight='balanced',
        max_iter=2000,
        random_state=42,
    )
    rfe_selector = RFE(
        estimator=rfe_estimator,
        n_features_to_select=args.n_features,
        step=1,
    )
    rfe_selector.fit(X_train, y_train)

    selected_mask  = rfe_selector.support_
    selected_names = np.array(feature_names)[selected_mask].tolist()
    print(f'  ✓ RFE complete. Selected features:')
    for i, name in enumerate(selected_names):
        print(f'    {i+1:2d}. {name}')

    # Transform to selected features
    X_train_sel = rfe_selector.transform(X_train)
    X_val_sel   = rfe_selector.transform(X_val)
    X_test_sel  = rfe_selector.transform(X_test)

    # ── Final classifier: RBF-SVM on selected features ─────────────────
    print('\nFitting final RBF-SVM on selected features...')
    final_clf = SVC(
        kernel='rbf',
        C=0.01,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42,
    )
    final_clf.fit(X_train_sel, y_train)
    print('  ✓ Classifier fitted')

    # ── Predictions ────────────────────────────────────────────────────
    prob_train = get_probs(final_clf, X_train_sel)
    prob_val   = get_probs(final_clf, X_val_sel)
    prob_test  = get_probs(final_clf, X_test_sel)

    pred_train = (prob_train >= 0.5).astype(int)
    pred_val   = (prob_val   >= 0.5).astype(int)
    pred_test  = (prob_test  >= 0.5).astype(int)

    # ── Metrics ────────────────────────────────────────────────────────
    m_train = compute_metrics(y_train, pred_train, prob_train, 'train')
    m_val   = compute_metrics(y_val,   pred_val,   prob_val,   'val')
    m_test  = compute_metrics(y_test,  pred_test,  prob_test,  'test')

    gap_tv = m_train['auc'] - m_val['auc']
    gap_tt = m_train['auc'] - m_test['auc']

    print(f'\n  {"Partition":8s} | {"AUC":>6} {"F1":>6} '
          f'{"Sens":>6} {"Spec":>6} {"Acc":>6} {"MCC":>6}')
    print(f'  {"-"*55}')
    for name, m in [('Train', m_train), ('Val', m_val), ('Test', m_test)]:
        print(f'  {name:8s} | {m.get("auc",0):6.3f} {m.get("f1",0):6.3f} '
              f'{m.get("sensitivity",0):6.3f} {m.get("specificity",0):6.3f} '
              f'{m.get("accuracy",0):6.3f} {m.get("mcc",0):6.3f}')

    flag = ' ⚠ OVERFIT' if gap_tt > 0.10 else ' ✓ OK'
    print(f'\n  Train-Val  gap : {gap_tv:.3f}')
    print(f'  Train-Test gap : {gap_tt:.3f}{flag}')

    # ── Plots ──────────────────────────────────────────────────────────
    print('\nGenerating plots...')

    plot_roc(y_val, prob_val, y_test, prob_test, output_dir)

    for y_p, y_pr, part in [
        (y_train, pred_train, 'train'),
        (y_val,   pred_val,   'val'),
        (y_test,  pred_test,  'test'),
    ]:
        plot_confusion_matrix(y_p, y_pr, part, output_dir)
        plot_calibration(y_p, get_probs(final_clf,
            X_train_sel if part == 'train' else
            X_val_sel   if part == 'val'   else X_test_sel),
            part, output_dir)

    plot_feature_stability(selected_mask, feature_names, output_dir)
    plot_per_patient_auc(y_test, prob_test, pat_test, output_dir)
    plot_overfitting_summary(m_train, m_val, m_test, output_dir)

    if not args.no_learning_curve:
        print('\n  Computing learning curve (use --no_learning_curve to skip)...')
        plot_learning_curve(rfe_selector, final_clf,
                            X_train, y_train, X_val, y_val, output_dir)

    # ── Save results ───────────────────────────────────────────────────
    results = {
        MODEL_NAME: {
            'train':           m_train,
            'val':             m_val,
            'test':            m_test,
            'selected_features': selected_names,
            'n_features_selected': int(args.n_features),
            'train_val_gap':   float(gap_tv),
            'train_test_gap':  float(gap_tt),
        }
    }

    out_path = output_dir / 'results_svm_rfe.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  ✓ Results → {out_path}')

    # ── Summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 65)
    print('STEP 4b COMPLETE — SVM-RFE')
    print('=' * 65)
    print(f'  Selected {args.n_features} features from {X.shape[1]}')
    print(f'  Test AUC  : {m_test["auc"]:.3f}')
    print(f'  Test MCC  : {m_test["mcc"]:.3f}')
    print(f'  Overfit gap (train-test): {gap_tt:+.3f}')
    print(f'\n  Results saved to: {out_path}')
    print(f'  Plots saved to  : {output_dir}')


if __name__ == '__main__':
    main()
