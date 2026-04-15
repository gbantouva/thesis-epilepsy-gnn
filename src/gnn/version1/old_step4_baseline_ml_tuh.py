"""
Step 4 — Baseline ML (TUH Dataset, Fixed 60-20-20 Split)
=========================================================
RF and SVM trained on the train split, hyperparameter selection on
val split (reported but NOT used for model selection — fixed HPs),
final evaluation on test split.

OVERFITTING ANALYSIS:
  Train AUC vs Test AUC gap is reported for both models.
  Feature importances (RF) shown for top-30 features.

Outputs:
  cm_<model>.png          confusion matrix (test set)
  roc_<model>.png         ROC curve (test set)
  feature_importance_rf.png
  overfitting_summary.png train vs val vs test AUC bar chart
  results_all.json
  summary_table.csv

Usage:
  python step4_baseline_ml.py \\
      --featfile  F:\\features\\features_all.npz \\
      --output_dir F:\\results\\baseline_ml
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              matthews_corrcoef, precision_score,
                              roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_total   = len(y_true)
    majority_n = max((y_true == 0).sum(), (y_true == 1).sum())
    return dict(
        accuracy         = float(accuracy_score(y_true, y_pred)),
        majority_baseline= float(majority_n / n_total),
        auc              = float(roc_auc_score(y_true, y_prob)),
        f1               = float(f1_score(y_true, y_pred, zero_division=0)),
        sensitivity      = float(tp / (tp + fn + 1e-12)),
        specificity      = float(tn / (tn + fp + 1e-12)),
        precision        = float(precision_score(y_true, y_pred, zero_division=0)),
        mcc              = float(matthews_corrcoef(y_true, y_pred)),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name, split_label, output_dir, cmap='Blues'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    for ax, data, fmt, title in [
        (axes[0], cm,      'd',   'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalised'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    xticklabels=['Control', 'Epilepsy'],
                    yticklabels=['Control', 'Epilepsy'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True',      fontsize=11)
        ax.set_title(f'{model_name} — {split_label} CM ({title})',
                     fontsize=11, fontweight='bold')
    plt.tight_layout()
    fname = output_dir / f'cm_{model_name.lower().replace(" ", "_")}_{split_label}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(fpr_val, tpr_val, auc_val,
             fpr_test, tpr_test, auc_test,
             model_name, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_val,  tpr_val,  lw=2, color='steelblue',
            label=f'Val  AUC={auc_val:.3f}')
    ax.plot(fpr_test, tpr_test, lw=2, color='tomato',
            label=f'Test AUC={auc_test:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title(f'{model_name} — ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importances, feature_names, output_dir, top_n=30):
    idx  = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(top_n), importances[idx],
           color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(np.array(feature_names)[idx], rotation=90, fontsize=7)
    ax.set_ylabel('Mean Decrease Impurity', fontsize=11)
    ax.set_title(f'Random Forest — Top {top_n} Feature Importances (Train set)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_rf.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  feature_importance_rf.png')


def plot_overfitting_summary(all_results, output_dir):
    """
    Bar chart: Train AUC vs Val AUC vs Test AUC for each model.
    Annotates the train–test gap to flag overfitting.
    """
    model_names = list(all_results.keys())
    train_aucs  = [all_results[m]['train_auc'] for m in model_names]
    val_aucs    = [all_results[m]['val_metrics']['auc']  for m in model_names]
    test_aucs   = [all_results[m]['test_metrics']['auc'] for m in model_names]

    x     = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, train_aucs, width, label='Train AUC',
           color='steelblue', alpha=0.85, edgecolor='black')
    ax.bar(x,         val_aucs,   width, label='Val AUC',
           color='seagreen',  alpha=0.85, edgecolor='black')
    ax.bar(x + width, test_aucs,  width, label='Test AUC',
           color='tomato',    alpha=0.85, edgecolor='black')

    for i, (tr, te) in enumerate(zip(train_aucs, test_aucs)):
        gap   = tr - te
        color = 'red' if gap > 0.10 else 'black'
        ax.text(i, max(tr, val_aucs[i], te) + 0.03,
                f'gap={gap:.2f}', ha='center', fontsize=10,
                fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.25)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax.set_title('Overfitting Analysis — Train / Val / Test AUC',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  overfitting_summary.png')


def plot_model_comparison(all_results, output_dir):
    met_keys = ['auc', 'f1', 'sensitivity', 'specificity']
    models   = list(all_results.keys())
    colors   = ['steelblue', 'tomato']
    x        = np.arange(len(met_keys))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, col) in enumerate(zip(models, colors)):
        vals = [all_results[name]['test_metrics'][k] for k in met_keys]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=name,
               color=col, alpha=0.85, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in met_keys], fontsize=12)
    ax.set_ylabel('Score (Test set)', fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Chance')
    ax.set_title('Baseline ML — Model Comparison (Test Set)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  model_comparison.png')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Step 4 — Baseline ML (TUH, fixed split)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--featfile',   required=True)
    parser.add_argument('--output_dir', default='results/baseline_ml')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 65)
    print('STEP 4 — BASELINE ML  (TUH, 60-20-20 fixed split)')
    print('=' * 65)

    # ── Load ──────────────────────────────────────────────────
    data          = np.load(args.featfile, allow_pickle=True)
    X             = data['X'].astype(np.float32)
    y             = data['y'].astype(np.int64)
    splits        = data['splits']
    feature_names = data['feature_names'].tolist()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    tr_mask = splits == 'train'
    va_mask = splits == 'val'
    te_mask = splits == 'test'

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_va, y_va = X[va_mask], y[va_mask]
    X_te, y_te = X[te_mask], y[te_mask]

    print(f'Train : {tr_mask.sum():,} epochs  '
          f'(epilepsy={int((y_tr==1).sum())}, control={int((y_tr==0).sum())})')
    print(f'Val   : {va_mask.sum():,} epochs  '
          f'(epilepsy={int((y_va==1).sum())}, control={int((y_va==0).sum())})')
    print(f'Test  : {te_mask.sum():,} epochs  '
          f'(epilepsy={int((y_te==1).sum())}, control={int((y_te==0).sum())})')
    majority_b = max((y_te == 0).sum(), (y_te == 1).sum()) / len(y_te)
    print(f'Majority-class accuracy baseline (test): {majority_b*100:.1f}%')

    # ── Scale — fit on TRAIN only ─────────────────────────────
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    # ── Models ────────────────────────────────────────────────
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        ),
        'SVM RBF': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42,
        ),
    }

    all_results = {}

    for model_name, clf in models.items():
        print(f'\n{"─"*55}')
        print(f'  {model_name}')
        print(f'{"─"*55}')

        clf.fit(X_tr_sc, y_tr)

        # Train AUC (overfitting check)
        if hasattr(clf, 'predict_proba'):
            prob_tr = clf.predict_proba(X_tr_sc)[:, 1]
        else:
            raw = clf.decision_function(X_tr_sc)
            prob_tr = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        train_auc = float(roc_auc_score(y_tr, prob_tr))

        results = {'train_auc': train_auc}

        for split_label, X_sc, y_sp in [
            ('val',  X_va_sc, y_va),
            ('test', X_te_sc, y_te),
        ]:
            if hasattr(clf, 'predict_proba'):
                y_prob = clf.predict_proba(X_sc)[:, 1]
            else:
                raw    = clf.decision_function(X_sc)
                y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
            y_pred   = clf.predict(X_sc)
            metrics  = compute_metrics(y_sp, y_pred, y_prob)
            results[f'{split_label}_metrics'] = metrics

            print(f'  {split_label.upper():5s}  AUC={metrics["auc"]:.3f}  '
                  f'F1={metrics["f1"]:.3f}  '
                  f'Sens={metrics["sensitivity"]:.3f}  '
                  f'Spec={metrics["specificity"]:.3f}  '
                  f'MCC={metrics["mcc"]:.3f}')

            cm = confusion_matrix(y_sp, y_pred)
            plot_confusion_matrix(cm, model_name, split_label, output_dir)

        # Gap
        gap = train_auc - results['test_metrics']['auc']
        flag = '  ⚠ OVERFIT' if gap > 0.10 else ''
        print(f'  Train AUC={train_auc:.3f}  '
              f'Test AUC={results["test_metrics"]["auc"]:.3f}  '
              f'Gap={gap:.3f}{flag}')

        # ROC
        #fpr_va, tpr_va, _ = roc_curve(y_va, results['val_metrics']['auc'])
        # recalculate properly
        if hasattr(clf, 'predict_proba'):
            p_va = clf.predict_proba(X_va_sc)[:, 1]
            p_te = clf.predict_proba(X_te_sc)[:, 1]
        else:
            raw_va = clf.decision_function(X_va_sc)
            raw_te = clf.decision_function(X_te_sc)
            p_va   = (raw_va - raw_va.min()) / (raw_va.max() - raw_va.min() + 1e-12)
            p_te   = (raw_te - raw_te.min()) / (raw_te.max() - raw_te.min() + 1e-12)
        fpr_va, tpr_va, _ = roc_curve(y_va, p_va)
        fpr_te, tpr_te, _ = roc_curve(y_te, p_te)
        plot_roc(fpr_va, tpr_va, results['val_metrics']['auc'],
                 fpr_te, tpr_te, results['test_metrics']['auc'],
                 model_name, output_dir)

        # Feature importance (RF only)
        if hasattr(clf, 'feature_importances_'):
            plot_feature_importance(clf.feature_importances_,
                                    feature_names, output_dir)

        all_results[model_name] = results

    # ── Cross-model plots ─────────────────────────────────────
    plot_overfitting_summary(all_results, output_dir)
    plot_model_comparison(all_results, output_dir)

    # ── Save JSON ─────────────────────────────────────────────
    results_path = output_dir / 'results_all.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  ✓ Saved: {results_path}')

    # ── Summary table ─────────────────────────────────────────
    rows = []
    for name, res in all_results.items():
        row = {'Model': name, 'Train AUC': f'{res["train_auc"]:.3f}'}
        for split_label in ('val', 'test'):
            m = res[f'{split_label}_metrics']
            for k in ('auc', 'f1', 'sensitivity', 'specificity', 'mcc'):
                row[f'{split_label}_{k}'] = f'{m[k]:.3f}'
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    print('\n' + df.to_string())
    df.to_csv(output_dir / 'summary_table.csv')

    print('\n' + '=' * 65)
    print('STEP 4 COMPLETE')
    print('=' * 65)


if __name__ == '__main__':
    main()