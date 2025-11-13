"""
Quick Test Script - Verify Connectivity Pipeline Works

This script runs a complete mini-test of your connectivity analysis pipeline:
1. Finds one epilepsy and one control patient
2. Runs connectivity analysis
3. Validates results
4. Creates comparison

Usage:
  python test_connectivity_pipeline.py \
    --data_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\test\\data_pp \
    --output_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\connectivity_test

This should take ~2-3 minutes and verifies everything is working correctly.
"""

import sys
from pathlib import Path
import argparse

# Import our scripts
from connectivity_analysis import process_file
from validate_connectivity import check_single_file
from compare_groups import find_connectivity_files, load_and_average_connectivity, plot_group_comparison


def find_test_patients(data_dir):
    """
    Find one epilepsy and one control patient for testing.
    
    Returns:
        (epilepsy_file, control_file) or (None, None) if not found
    """
    data_dir = Path(data_dir)
    
    # Find all epoch files
    all_epochs = list(data_dir.rglob("*_epochs.npy"))
    
    epilepsy_file = None
    control_file = None
    
    for epoch_file in all_epochs:
        path_str = str(epoch_file).replace('\\', '/')
        
        if epilepsy_file is None and '/00_epilepsy/' in path_str:
            epilepsy_file = epoch_file
        
        if control_file is None and ('/01_no_epilepsy/' in path_str or '/01_control/' in path_str):
            control_file = epoch_file
        
        if epilepsy_file and control_file:
            break
    
    return epilepsy_file, control_file


def main():
    parser = argparse.ArgumentParser(
        description="Quick test of connectivity analysis pipeline"
    )
    
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Preprocessed data directory"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for test results"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("CONNECTIVITY PIPELINE TEST")
    print("="*70)
    print("\nThis test will:")
    print("  1. Find one epilepsy and one control patient")
    print("  2. Compute DTF and PDC connectivity")
    print("  3. Validate results")
    print("  4. Create comparison visualization")
    print("\nEstimated time: 2-3 minutes")
    print("="*70 + "\n")
    
    # Step 1: Find test patients
    print("Step 1: Finding test patients...")
    epilepsy_file, control_file = find_test_patients(data_dir)
    
    if epilepsy_file is None:
        print("ERROR: No epilepsy patient found in data_dir")
        print("Expected path: */00_epilepsy/*")
        sys.exit(1)
    
    if control_file is None:
        print("ERROR: No control patient found in data_dir")
        print("Expected path: */01_no_epilepsy/* or */01_control/*")
        sys.exit(1)
    
    print(f"✓ Found epilepsy patient: {epilepsy_file.stem}")
    print(f"✓ Found control patient: {control_file.stem}")
    
    # Step 2: Run connectivity analysis
    print("\n" + "="*70)
    print("Step 2: Computing connectivity...")
    print("="*70)
    
    # Process epilepsy patient
    print("\nProcessing epilepsy patient...")
    epi_output_dir = output_dir / "epilepsy"
    epi_success = process_file(
        epoch_file=epilepsy_file,
        output_dir=epi_output_dir,
        model_order=None  # Auto-select
    )
    
    if not epi_success:
        print("ERROR: Failed to process epilepsy patient")
        sys.exit(1)
    
    print("✓ Epilepsy patient processed successfully")
    
    # Process control patient
    print("\nProcessing control patient...")
    ctrl_output_dir = output_dir / "control"
    ctrl_success = process_file(
        epoch_file=control_file,
        output_dir=ctrl_output_dir,
        model_order=None
    )
    
    if not ctrl_success:
        print("ERROR: Failed to process control patient")
        sys.exit(1)
    
    print("✓ Control patient processed successfully")
    
    # Step 3: Validate results
    print("\n" + "="*70)
    print("Step 3: Validating results...")
    print("="*70)
    
    # Find output files
    epi_pid = epilepsy_file.stem.replace("_epochs", "")
    ctrl_pid = control_file.stem.replace("_epochs", "")
    
    # Validate epilepsy
    print(f"\nValidating epilepsy patient ({epi_pid})...")
    epi_result = check_single_file(
        dtf_file=epi_output_dir / f"{epi_pid}_dtf_alpha.npy",
        pdc_file=epi_output_dir / f"{epi_pid}_pdc_alpha.npy",
        metadata_file=epi_output_dir / f"{epi_pid}_connectivity_metadata.json"
    )
    
    if epi_result.get('status') != 'OK':
        print("WARNING: Epilepsy patient validation found issues:")
        for issue in epi_result.get('issues', []):
            print(f"  - {issue}")
    else:
        print("✓ Epilepsy patient validation passed")
        print(f"  DTF mean: {epi_result['metrics']['dtf_mean']:.4f}")
        print(f"  PDC mean: {epi_result['metrics']['pdc_mean']:.4f}")
    
    # Validate control
    print(f"\nValidating control patient ({ctrl_pid})...")
    ctrl_result = check_single_file(
        dtf_file=ctrl_output_dir / f"{ctrl_pid}_dtf_alpha.npy",
        pdc_file=ctrl_output_dir / f"{ctrl_pid}_pdc_alpha.npy",
        metadata_file=ctrl_output_dir / f"{ctrl_pid}_connectivity_metadata.json"
    )
    
    if ctrl_result.get('status') != 'OK':
        print("WARNING: Control patient validation found issues:")
        for issue in ctrl_result.get('issues', []):
            print(f"  - {issue}")
    else:
        print("✓ Control patient validation passed")
        print(f"  DTF mean: {ctrl_result['metrics']['dtf_mean']:.4f}")
        print(f"  PDC mean: {ctrl_result['metrics']['pdc_mean']:.4f}")
    
    # Step 4: Quick comparison
    print("\n" + "="*70)
    print("Step 4: Creating comparison visualization...")
    print("="*70)
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load DTF alpha for both patients
    epi_dtf = np.load(epi_output_dir / f"{epi_pid}_dtf_alpha.npy")
    ctrl_dtf = np.load(ctrl_output_dir / f"{ctrl_pid}_dtf_alpha.npy")
    
    # Average over epochs
    epi_mean = epi_dtf.mean(axis=0)
    ctrl_mean = ctrl_dtf.mean(axis=0)
    diff = epi_mean - ctrl_mean
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Channel names
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
                "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"]
    
    tick_indices = np.arange(0, len(ch_names), 5)
    
    # Epilepsy
    im0 = axes[0].imshow(epi_mean, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[0].set_title(f"EPILEPSY\n{epi_pid}", fontweight='bold', color='red')
    axes[0].set_xlabel("Source Channel")
    axes[0].set_ylabel("Target Channel")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].set_xticks(tick_indices)
    axes[0].set_yticks(tick_indices)
    axes[0].set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels([ch_names[i] for i in tick_indices], fontsize=8)
    
    # Control
    im1 = axes[1].imshow(ctrl_mean, cmap='hot', vmin=0, vmax=1, aspect='auto')
    axes[1].set_title(f"CONTROL\n{ctrl_pid}", fontweight='bold', color='blue')
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
    axes[2].set_title("DIFFERENCE\n(Epilepsy - Control)", fontweight='bold')
    axes[2].set_xlabel("Source Channel")
    axes[2].set_ylabel("Target Channel")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label='Difference')
    axes[2].set_xticks(tick_indices)
    axes[2].set_yticks(tick_indices)
    axes[2].set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
    axes[2].set_yticklabels([ch_names[i] for i in tick_indices], fontsize=8)
    
    plt.suptitle("TEST COMPARISON: DTF Alpha Band", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = output_dir / "test_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison saved to: {comparison_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print("\n✓ All steps completed successfully!")
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - epilepsy/{epi_pid}_connectivity_visualization.png")
    print(f"  - control/{ctrl_pid}_connectivity_visualization.png")
    print(f"  - test_comparison.png")
    print("\nWhat to check:")
    print("  1. Open the visualization files - do you see patterns (not all black/white)?")
    print("  2. Open test_comparison.png - does the difference plot show structure?")
    print("  3. If both look good, your pipeline is working!")
    print("\nNext steps:")
    print("  1. Run connectivity_batch.py to process all patients")
    print("  2. Run validate_connectivity.py to check quality")
    print("  3. Run compare_groups.py for full statistical comparison")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()