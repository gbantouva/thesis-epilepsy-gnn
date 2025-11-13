"""
Connectivity Validation Script

This script helps you verify your connectivity results are correct.
It checks for common issues and provides quality metrics.

Usage:
  python validate_connectivity.py --connectivity_dir connectivity_results

Example:
  python validate_connectivity.py \
    --connectivity_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\connectivity_results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from collections import defaultdict


def check_single_file(dtf_file, pdc_file, metadata_file):
    """
    Validate a single patient's connectivity results.
    
    Returns dictionary with quality metrics.
    """
    issues = []
    warnings = []
    metrics = {}
    
    # Load data
    try:
        dtf = np.load(dtf_file)
        pdc = np.load(pdc_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        return {"error": str(e)}
    
    # Check shapes
    if dtf.shape != pdc.shape:
        issues.append(f"Shape mismatch: DTF={dtf.shape}, PDC={pdc.shape}")
    
    expected_shape = (metadata['n_epochs'], metadata['n_channels'], metadata['n_channels'])
    if dtf.shape != expected_shape:
        issues.append(f"Unexpected shape: got {dtf.shape}, expected {expected_shape}")
    
    # Check value ranges (should be 0 to 1)
    if dtf.min() < 0 or dtf.max() > 1:
        issues.append(f"DTF values out of range: [{dtf.min():.3f}, {dtf.max():.3f}]")
    
    if pdc.min() < 0 or pdc.max() > 1:
        issues.append(f"PDC values out of range: [{pdc.min():.3f}, {pdc.max():.3f}]")
    
    # Check for NaN or Inf
    if np.any(np.isnan(dtf)) or np.any(np.isinf(dtf)):
        issues.append("DTF contains NaN or Inf values")
    
    if np.any(np.isnan(pdc)) or np.any(np.isinf(pdc)):
        issues.append("PDC contains NaN or Inf values")
    
    # Check diagonal (self-connections should be low)
    dtf_diag = np.array([dtf[i, np.arange(22), np.arange(22)].mean() for i in range(dtf.shape[0])])
    pdc_diag = np.array([pdc[i, np.arange(22), np.arange(22)].mean() for i in range(pdc.shape[0])])
    
    if dtf_diag.mean() > 0.5:
        warnings.append(f"High diagonal values in DTF (mean={dtf_diag.mean():.3f}), expected < 0.5")
    
    if pdc_diag.mean() > 0.5:
        warnings.append(f"High diagonal values in PDC (mean={pdc_diag.mean():.3f}), expected < 0.5")
    
    # Check variance (should not be all zeros or all ones)
    if dtf.std() < 0.01:
        warnings.append(f"Very low DTF variance ({dtf.std():.4f}), data might be too uniform")
    
    if pdc.std() < 0.01:
        warnings.append(f"Very low PDC variance ({pdc.std():.4f}), data might be too uniform")
    
    # Compute metrics
    metrics['dtf_mean'] = float(dtf.mean())
    metrics['dtf_std'] = float(dtf.std())
    metrics['dtf_min'] = float(dtf.min())
    metrics['dtf_max'] = float(dtf.max())
    
    metrics['pdc_mean'] = float(pdc.mean())
    metrics['pdc_std'] = float(pdc.std())
    metrics['pdc_min'] = float(pdc.min())
    metrics['pdc_max'] = float(pdc.max())
    
    metrics['diagonal_dtf_mean'] = float(dtf_diag.mean())
    metrics['diagonal_pdc_mean'] = float(pdc_diag.mean())
    
    metrics['n_epochs'] = metadata['n_epochs']
    metrics['avg_model_order'] = metadata['avg_model_order']
    
    return {
        'metrics': metrics,
        'issues': issues,
        'warnings': warnings,
        'status': 'OK' if len(issues) == 0 else 'FAILED'
    }


def find_connectivity_files(connectivity_dir):
    """
    Find all connectivity result files.
    
    Returns list of (dtf_alpha, pdc_alpha, metadata) tuples.
    """
    connectivity_dir = Path(connectivity_dir)
    
    # Find all metadata files (one per patient)
    metadata_files = list(connectivity_dir.rglob("*_connectivity_metadata.json"))
    
    results = []
    
    for metadata_file in metadata_files:
        # Extract patient ID
        pid = metadata_file.stem.replace("_connectivity_metadata", "")
        
        # Find corresponding DTF and PDC files (alpha band as representative)
        dtf_file = metadata_file.parent / f"{pid}_dtf_alpha.npy"
        pdc_file = metadata_file.parent / f"{pid}_pdc_alpha.npy"
        
        if dtf_file.exists() and pdc_file.exists():
            results.append((dtf_file, pdc_file, metadata_file))
    
    return results


def create_summary_report(all_results, output_dir):
    """
    Create a summary report of all validation results.
    """
    output_dir = Path(output_dir)
    
    # Separate by status
    ok_patients = [r for r in all_results if r['result']['status'] == 'OK']
    failed_patients = [r for r in all_results if r['result']['status'] == 'FAILED']
    
    # Create report
    report_path = output_dir / "validation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CONNECTIVITY VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Total patients: {len(all_results)}\n")
        f.write(f"  Passed: {len(ok_patients)} ({100*len(ok_patients)/max(len(all_results),1):.1f}%)\n")
        f.write(f"  Failed: {len(failed_patients)} ({100*len(failed_patients)/max(len(all_results),1):.1f}%)\n\n")
        
        if ok_patients:
            # Compute aggregate statistics
            dtf_means = [r['result']['metrics']['dtf_mean'] for r in ok_patients]
            pdc_means = [r['result']['metrics']['pdc_mean'] for r in ok_patients]
            model_orders = [r['result']['metrics']['avg_model_order'] for r in ok_patients]
            
            f.write("AGGREGATE STATISTICS (PASSED PATIENTS):\n")
            f.write(f"  DTF mean: {np.mean(dtf_means):.3f} ± {np.std(dtf_means):.3f}\n")
            f.write(f"  PDC mean: {np.mean(pdc_means):.3f} ± {np.std(pdc_means):.3f}\n")
            f.write(f"  Model order: {np.mean(model_orders):.1f} ± {np.std(model_orders):.1f}\n\n")
        
        if failed_patients:
            f.write("FAILED PATIENTS:\n")
            for r in failed_patients[:10]:  # Show first 10
                f.write(f"\n  {r['patient_id']}:\n")
                for issue in r['result']['issues']:
                    f.write(f"    - {issue}\n")
            
            if len(failed_patients) > 10:
                f.write(f"\n  ... and {len(failed_patients) - 10} more\n")
        
        # Warnings
        all_warnings = []
        for r in ok_patients:
            if r['result']['warnings']:
                all_warnings.extend([(r['patient_id'], w) for w in r['result']['warnings']])
        
        if all_warnings:
            f.write("\n" + "="*70 + "\n")
            f.write("WARNINGS (first 20):\n")
            for pid, warning in all_warnings[:20]:
                f.write(f"  {pid}: {warning}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n✓ Report saved to: {report_path}")
    
    return report_path


def plot_quality_metrics(all_results, output_dir):
    """
    Create visualization of quality metrics.
    """
    output_dir = Path(output_dir)
    
    # Extract data
    ok_results = [r for r in all_results if r['result']['status'] == 'OK']
    
    if not ok_results:
        print("No valid results to plot")
        return
    
    dtf_means = [r['result']['metrics']['dtf_mean'] for r in ok_results]
    pdc_means = [r['result']['metrics']['pdc_mean'] for r in ok_results]
    dtf_stds = [r['result']['metrics']['dtf_std'] for r in ok_results]
    pdc_stds = [r['result']['metrics']['pdc_std'] for r in ok_results]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # DTF mean distribution
    axes[0, 0].hist(dtf_means, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(dtf_means), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(dtf_means):.3f}')
    axes[0, 0].set_xlabel("DTF Mean Value", fontweight='bold')
    axes[0, 0].set_ylabel("Number of Patients", fontweight='bold')
    axes[0, 0].set_title("DTF Mean Distribution", fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # PDC mean distribution
    axes[0, 1].hist(pdc_means, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(pdc_means), color='red', linestyle='--',
                       label=f'Mean: {np.mean(pdc_means):.3f}')
    axes[0, 1].set_xlabel("PDC Mean Value", fontweight='bold')
    axes[0, 1].set_ylabel("Number of Patients", fontweight='bold')
    axes[0, 1].set_title("PDC Mean Distribution", fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # DTF std distribution
    axes[1, 0].hist(dtf_stds, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(dtf_stds), color='red', linestyle='--',
                       label=f'Mean: {np.mean(dtf_stds):.3f}')
    axes[1, 0].set_xlabel("DTF Std Dev", fontweight='bold')
    axes[1, 0].set_ylabel("Number of Patients", fontweight='bold')
    axes[1, 0].set_title("DTF Variance Distribution", fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # PDC std distribution
    axes[1, 1].hist(pdc_stds, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(pdc_stds), color='red', linestyle='--',
                       label=f'Mean: {np.mean(pdc_stds):.3f}')
    axes[1, 1].set_xlabel("PDC Std Dev", fontweight='bold')
    axes[1, 1].set_ylabel("Number of Patients", fontweight='bold')
    axes[1, 1].set_title("PDC Variance Distribution", fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle("Connectivity Quality Metrics", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / "quality_metrics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Quality metrics plot saved to: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate connectivity analysis results"
    )
    
    parser.add_argument(
        "--connectivity_dir",
        required=True,
        help="Directory containing connectivity results"
    )
    
    args = parser.parse_args()
    
    connectivity_dir = Path(args.connectivity_dir)
    
    if not connectivity_dir.exists():
        print(f"ERROR: Directory not found: {connectivity_dir}")
        return
    
    print("\n" + "="*70)
    print("CONNECTIVITY VALIDATION")
    print("="*70)
    
    # Find all connectivity files
    print("\nSearching for connectivity results...")
    files = find_connectivity_files(connectivity_dir)
    
    if len(files) == 0:
        print(f"ERROR: No connectivity results found in {connectivity_dir}")
        print("Expected files: *_dtf_alpha.npy, *_pdc_alpha.npy, *_connectivity_metadata.json")
        return
    
    print(f"Found {len(files)} patients with connectivity results")
    
    # Validate each patient
    print("\nValidating results...")
    all_results = []
    
    from tqdm import tqdm
    
    for dtf_file, pdc_file, metadata_file in tqdm(files, desc="Validating"):
        pid = metadata_file.stem.replace("_connectivity_metadata", "")
        
        result = check_single_file(dtf_file, pdc_file, metadata_file)
        
        all_results.append({
            'patient_id': pid,
            'result': result
        })
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    ok_count = sum(1 for r in all_results if r['result']['status'] == 'OK')
    failed_count = sum(1 for r in all_results if r['result']['status'] == 'FAILED')
    
    print(f"Total patients: {len(all_results)}")
    print(f"Passed: {ok_count} ({100*ok_count/len(all_results):.1f}%)")
    print(f"Failed: {failed_count} ({100*failed_count/len(all_results):.1f}%)")
    
    if ok_count > 0:
        # Show some example metrics
        example = next(r for r in all_results if r['result']['status'] == 'OK')
        print(f"\nExample metrics (patient {example['patient_id']}):")
        for key, value in example['result']['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Create detailed report
    report_path = create_summary_report(all_results, connectivity_dir)
    
    # Create quality plots
    if ok_count > 0:
        plot_quality_metrics(all_results, connectivity_dir)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nDetailed report: {report_path}")
    print("\nNext steps:")
    print("  1. Review the validation report")
    print("  2. Check quality_metrics.png for distributions")
    print("  3. If most patients passed, you're ready for graph construction!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
