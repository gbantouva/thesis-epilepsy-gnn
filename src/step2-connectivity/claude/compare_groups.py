"""
Quick Connectivity Comparison: Epilepsy vs Control

This script helps you visually compare connectivity patterns between groups.
Use this to verify your connectivity analysis captures meaningful differences.

Usage:
  python compare_groups.py --connectivity_dir connectivity_results --output_dir comparison_results

Example:
  python compare_groups.py \
    --connectivity_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\connectivity_results \
    --output_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\comparison_results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
from tqdm import tqdm


def find_connectivity_files(connectivity_dir):
    """
    Find connectivity files and separate by group.
    
    Returns:
        (epilepsy_files, control_files) - lists of (patient_id, dtf_files, pdc_files)
    """
    connectivity_dir = Path(connectivity_dir)
    
    epilepsy_files = []
    control_files = []
    
    # Find all DTF alpha files
    dtf_alpha_files = list(connectivity_dir.rglob("*_dtf_alpha.npy"))
    
    for dtf_file in dtf_alpha_files:
        # Determine group from path
        path_str = str(dtf_file).replace('\\', '/')
        
        if '/00_epilepsy/' in path_str or '\\00_epilepsy\\' in str(dtf_file):
            group = 'epilepsy'
        elif '/01_no_epilepsy/' in path_str or '\\01_no_epilepsy\\' in str(dtf_file):
            group = 'control'
        elif '/01_control/' in path_str or '\\01_control\\' in str(dtf_file):
            group = 'control'
        else:
            print(f"Warning: Cannot determine group for {dtf_file}")
            continue
        
        # Get patient ID
        pid = dtf_file.stem.replace("_dtf_alpha", "")
        
        # Find all band files
        dtf_files = {
            'delta': dtf_file.parent / f"{pid}_dtf_delta.npy",
            'theta': dtf_file.parent / f"{pid}_dtf_theta.npy",
            'alpha': dtf_file.parent / f"{pid}_dtf_alpha.npy",
            'beta': dtf_file.parent / f"{pid}_dtf_beta.npy",
            'gamma': dtf_file.parent / f"{pid}_dtf_gamma.npy"
        }
        
        pdc_files = {
            'delta': dtf_file.parent / f"{pid}_pdc_delta.npy",
            'theta': dtf_file.parent / f"{pid}_pdc_theta.npy",
            'alpha': dtf_file.parent / f"{pid}_pdc_alpha.npy",
            'beta': dtf_file.parent / f"{pid}_pdc_beta.npy",
            'gamma': dtf_file.parent / f"{pid}_pdc_gamma.npy"
        }
        
        # Check all files exist
        if all(f.exists() for f in dtf_files.values()) and all(f.exists() for f in pdc_files.values()):
            if group == 'epilepsy':
                epilepsy_files.append((pid, dtf_files, pdc_files))
            else:
                control_files.append((pid, dtf_files, pdc_files))
    
    return epilepsy_files, control_files


def load_and_average_connectivity(files, measure='dtf', band='alpha', max_patients=None):
    """
    Load connectivity files and compute per-patient averages.
    
    Args:
        files: List of (patient_id, dtf_files, pdc_files)
        measure: 'dtf' or 'pdc'
        band: 'delta', 'theta', 'alpha', 'beta', or 'gamma'
        max_patients: Maximum number of patients to load (None = all)
    
    Returns:
        (patient_ids, avg_connectivity) where avg_connectivity is (n_patients, 22, 22)
    """
    if max_patients:
        files = files[:max_patients]
    
    patient_ids = []
    avg_matrices = []
    
    for pid, dtf_files, pdc_files in tqdm(files, desc=f"Loading {measure} {band}"):
        try:
            # Select file
            if measure == 'dtf':
                file_path = dtf_files[band]
            else:
                file_path = pdc_files[band]
            
            # Load connectivity
            connectivity = np.load(file_path)  # Shape: (n_epochs, 22, 22)
            
            # Average over epochs
            avg_conn = connectivity.mean(axis=0)  # Shape: (22, 22)
            
            patient_ids.append(pid)
            avg_matrices.append(avg_conn)
        
        except Exception as e:
            print(f"Warning: Failed to load {pid}: {e}")
            continue
    
    if not avg_matrices:
        return None, None
    
    return patient_ids, np.array(avg_matrices)


def plot_group_comparison(epilepsy_conn, control_conn, ch_names, measure, band, output_dir):
    """
    Create comparison plot: Epilepsy vs Control.
    """
    # Compute group means
    epi_mean = epilepsy_conn.mean(axis=0)  # (22, 22)
    ctrl_mean = control_conn.mean(axis=0)  # (22, 22)
    
    # Compute difference
    diff = epi_mean - ctrl_mean
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Epilepsy
    im0 = axes[0].imshow(epi_mean, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f"EPILEPSY (n={len(epilepsy_conn)})\n{measure.upper()} - {band.upper()}", 
                     fontweight='bold', color='red')
    axes[0].set_xlabel("Source Channel")
    axes[0].set_ylabel("Target Channel")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Set channel labels (every 5th)
    tick_indices = np.arange(0, len(ch_names), 5)
    axes[0].set_xticks(tick_indices)
    axes[0].set_yticks(tick_indices)
    axes[0].set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels([ch_names[i] for i in tick_indices], fontsize=8)
    
    # Control
    im1 = axes[1].imshow(ctrl_mean, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f"CONTROL (n={len(control_conn)})\n{measure.upper()} - {band.upper()}", 
                     fontweight='bold', color='blue')
    axes[1].set_xlabel("Source Channel")
    axes[1].set_ylabel("Target Channel")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    axes[1].set_xticks(tick_indices)
    axes[1].set_yticks(tick_indices)
    axes[1].set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels([ch_names[i] for i in tick_indices], fontsize=8)
    
    # Difference
    max_abs = max(abs(diff.min()), abs(diff.max()))
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, aspect='auto')
    axes[2].set_title(f"DIFFERENCE (Epilepsy - Control)\n{measure.upper()} - {band.upper()}", 
                     fontweight='bold')
    axes[2].set_xlabel("Source Channel")
    axes[2].set_ylabel("Target Channel")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label='Difference')
    
    axes[2].set_xticks(tick_indices)
    axes[2].set_yticks(tick_indices)
    axes[2].set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels([ch_names[i] for i in tick_indices], fontsize=8)
    
    plt.tight_layout()
    
    # Save
    fig_path = output_dir / f"{measure}_{band}_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def compute_global_connectivity_stats(epilepsy_conn, control_conn):
    """
    Compute global connectivity strength for each patient.
    
    Returns:
        (epi_strengths, ctrl_strengths, t_stat, p_value)
    """
    # Global connectivity = mean of all connections (excluding diagonal)
    mask = ~np.eye(epilepsy_conn.shape[1], dtype=bool)
    
    epi_strengths = np.array([conn[mask].mean() for conn in epilepsy_conn])
    ctrl_strengths = np.array([conn[mask].mean() for conn in control_conn])
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(epi_strengths, ctrl_strengths)
    
    return epi_strengths, ctrl_strengths, t_stat, p_value


def plot_global_connectivity_comparison(epilepsy_files, control_files, output_dir):
    """
    Compare global connectivity strength across all bands.
    """
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    measures = ['dtf', 'pdc']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for measure_idx, measure in enumerate(measures):
        for band_idx, band in enumerate(bands):
            ax = axes[measure_idx, band_idx]
            
            # Load data
            _, epi_conn = load_and_average_connectivity(epilepsy_files, measure, band)
            _, ctrl_conn = load_and_average_connectivity(control_files, measure, band)
            
            if epi_conn is None or ctrl_conn is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Compute global strengths
            epi_strengths, ctrl_strengths, t_stat, p_value = compute_global_connectivity_stats(epi_conn, ctrl_conn)
            
            # Box plot
            data = [epi_strengths, ctrl_strengths]
            bp = ax.boxplot(data, labels=['Epilepsy', 'Control'], patch_artist=True)
            
            # Color boxes
            bp['boxes'][0].set_facecolor('red')
            bp['boxes'][0].set_alpha(0.5)
            bp['boxes'][1].set_facecolor('blue')
            bp['boxes'][1].set_alpha(0.5)
            
            # Add significance marker
            if p_value < 0.001:
                sig_marker = '***'
            elif p_value < 0.01:
                sig_marker = '**'
            elif p_value < 0.05:
                sig_marker = '*'
            else:
                sig_marker = 'ns'
            
            y_max = max(epi_strengths.max(), ctrl_strengths.max())
            ax.text(1.5, y_max * 1.1, sig_marker, ha='center', fontsize=12, fontweight='bold')
            
            # Title
            ax.set_title(f"{measure.upper()} - {band.upper()}\np={p_value:.4f}", fontsize=10)
            ax.set_ylabel("Global Connectivity Strength")
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Global Connectivity Comparison: Epilepsy vs Control\n(* p<0.05, ** p<0.01, *** p<0.001)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig_path = output_dir / "global_connectivity_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare connectivity between epilepsy and control groups"
    )
    
    parser.add_argument(
        "--connectivity_dir",
        required=True,
        help="Directory containing connectivity results"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for comparison plots"
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Maximum patients per group (for testing)"
    )
    
    args = parser.parse_args()
    
    connectivity_dir = Path(args.connectivity_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CONNECTIVITY GROUP COMPARISON")
    print("="*70)
    
    # Find files
    print("\nSearching for connectivity files...")
    epilepsy_files, control_files = find_connectivity_files(connectivity_dir)
    
    if not epilepsy_files:
        print("ERROR: No epilepsy patients found!")
        print("Expected path pattern: */00_epilepsy/*")
        return
    
    if not control_files:
        print("ERROR: No control patients found!")
        print("Expected path pattern: */01_no_epilepsy/* or */01_control/*")
        return
    
    print(f"Found {len(epilepsy_files)} epilepsy patients")
    print(f"Found {len(control_files)} control patients")
    
    # Limit if requested
    if args.max_patients:
        epilepsy_files = epilepsy_files[:args.max_patients]
        control_files = control_files[:args.max_patients]
        print(f"Limited to {len(epilepsy_files)} epilepsy and {len(control_files)} control patients")
    
    # Channel names (assumed to be the same for all)
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
                "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"]
    
    # Create comparison plots for each measure and band
    print("\nGenerating comparison plots...")
    
    measures = ['dtf', 'pdc']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for measure in measures:
        for band in bands:
            print(f"\n  Processing {measure.upper()} - {band.upper()}...")
            
            # Load data
            _, epi_conn = load_and_average_connectivity(epilepsy_files, measure, band, args.max_patients)
            _, ctrl_conn = load_and_average_connectivity(control_files, measure, band, args.max_patients)
            
            if epi_conn is None or ctrl_conn is None:
                print(f"    ✗ Failed to load data")
                continue
            
            # Create plot
            fig_path = plot_group_comparison(epi_conn, ctrl_conn, ch_names, measure, band, output_dir)
            print(f"    ✓ Saved: {fig_path.name}")
    
    # Create global comparison plot
    print("\n  Creating global connectivity comparison...")
    global_fig = plot_global_connectivity_comparison(epilepsy_files, control_files, output_dir)
    print(f"    ✓ Saved: {global_fig.name}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - *_comparison.png : Detailed matrix comparisons (10 files)")
    print("  - global_connectivity_comparison.png : Statistical overview")
    print("\nWhat to look for:")
    print("  ✓ Difference matrices should show patterns (not random)")
    print("  ✓ Statistical tests (p-values) indicate group differences")
    print("  ✓ Red/blue colors in difference plot = epilepsy has stronger/weaker connections")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
