"""
STEP 1B: EPOCH-LEVEL BIC HISTOGRAM
==================================
Lightweight script to collect optimal orders at the EPOCH level
for visualization purposes.

This complements step1_comprehensive_bic_analysis.py by adding
the epoch-level histogram to complete the hierarchical picture.

Usage:
    # Full analysis (all epochs - may take a while)
    python step1b_epoch_level_histogram.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced

    # Sampled analysis (faster, 100 epochs per file)
    python step1b_epoch_level_histogram.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced \
        --sample_size 100

    # Quick test (10 epochs per file)
    python step1b_epoch_level_histogram.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced \
        --sample_size 10
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTS
# ==============================================================================
MIN_ORDER = 5
MAX_ORDER = 18
RECOMMENDED_ORDER = 13  # From your main analysis


# ==============================================================================
# BIC COMPUTATION (same as main script)
# ==============================================================================

def compute_bic_for_epoch(data, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """
    Compute BIC curve for a single epoch and return optimal order.
    """
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    data_scaled = data / data_std
    
    try:
        model = VAR(data_scaled.T)
        
        bic_values = []
        for p in range(min_order, max_order + 1):
            try:
                result = model.fit(maxlags=p, trend='c', verbose=False)
                bic = result.bic
                
                if not np.isnan(bic) and not np.isinf(bic):
                    bic_values.append((p, bic))
            except:
                continue
        
        if len(bic_values) == 0:
            return None
        
        best_order, _ = min(bic_values, key=lambda x: x[1])
        return best_order
        
    except:
        return None


def process_file_epochs(args):
    """
    Process one file and return ALL epoch orders (not just mean).
    """
    epoch_file, sample_size, min_order, max_order = args
    
    try:
        epochs = np.load(epoch_file)
        n_epochs = len(epochs)
        
        # Sample epochs if requested
        if sample_size is not None and sample_size < n_epochs:
            indices = np.random.choice(n_epochs, sample_size, replace=False)
        else:
            indices = np.arange(n_epochs)
        
        # Extract group from path
        path_str = str(epoch_file)
        if '00_epilepsy' in path_str:
            group = 'epilepsy'
        elif '01_no_epilepsy' in path_str:
            group = 'control'
        else:
            group = 'unknown'
        
        # Collect orders for each epoch
        results = []
        for idx in indices:
            order = compute_bic_for_epoch(epochs[idx], min_order, max_order)
            if order is not None:
                results.append({
                    'file': epoch_file.name,
                    'epoch_idx': int(idx),
                    'optimal_order': order,
                    'group': group
                })
        
        return results
        
    except Exception as e:
        return []


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def collect_epoch_orders(input_dir, sample_size=None, n_workers=None):
    """
    Collect optimal orders at epoch level from all files.
    """
    input_path = Path(input_dir)
    
    # Find all epoch files
    epoch_files = list(input_path.rglob("*_epochs.npy"))
    print(f"📁 Found {len(epoch_files)} epoch files")
    
    if len(epoch_files) == 0:
        print("❌ No epoch files found!")
        return None
    
    # Prepare arguments for parallel processing
    args_list = [(f, sample_size, MIN_ORDER, MAX_ORDER) for f in epoch_files]
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"🔧 Using {n_workers} parallel workers")
    print(f"📊 Sample size per file: {'ALL' if sample_size is None else sample_size}")
    print(f"📈 Order range: {MIN_ORDER}-{MAX_ORDER}")
    print()
    
    # Process files in parallel
    all_results = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = list(tqdm(
            executor.map(process_file_epochs, args_list),
            total=len(args_list),
            desc="Processing files"
        ))
        
        for result_list in futures:
            all_results.extend(result_list)
    
    print(f"\n✅ Collected {len(all_results)} epoch-level orders")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    return df


def create_epoch_histogram(df, recommended_order=RECOMMENDED_ORDER):
    """
    Create epoch-level histogram visualization.
    """
    print("\n📊 Creating visualizations...")
    
    # =========================================================================
    # Figure 1: Simple epoch-level histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    orders = df['optimal_order'].values
    
    # Create histogram
    counts, bins, patches = ax.hist(
        orders, 
        bins=range(MIN_ORDER, MAX_ORDER + 2),  # +2 to include MAX_ORDER edge
        edgecolor='black', 
        alpha=0.7, 
        color='steelblue',
        align='left'
    )
    
    # Mark recommended order
    ax.axvline(recommended_order, color='red', linestyle='--', linewidth=2.5, 
               label=f'Recommended: p={recommended_order}')
    
    # Add count labels on top of bars
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width()/2, count + max(counts)*0.01,
                   f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    # Labels and title
    ax.set_xlabel('Optimal Order (p)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count (number of epochs)', fontsize=14, fontweight='bold')
    ax.set_title(f'Epoch-Level BIC Distribution (n={len(orders):,} epochs)', 
                fontsize=16, fontweight='bold')
    
    # Set x-ticks to show all orders
    ax.set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
    
    # Add statistics text box
    stats_text = f"Statistics:\n"
    stats_text += f"Total epochs: {len(orders):,}\n"
    stats_text += f"Mean: {np.mean(orders):.2f}\n"
    stats_text += f"Median: {np.median(orders):.1f}\n"
    stats_text += f"Std: {np.std(orders):.2f}\n"
    stats_text += f"Mode: {int(pd.Series(orders).mode()[0])}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('step1b_epoch_level_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: step1b_epoch_level_histogram.png")
    
    # =========================================================================
    # Figure 2: Epoch-level histogram by group
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    epilepsy_orders = df[df['group'] == 'epilepsy']['optimal_order'].values
    control_orders = df[df['group'] == 'control']['optimal_order'].values
    
    # Create grouped histogram
    bins = range(MIN_ORDER, MAX_ORDER + 2)
    ax.hist([epilepsy_orders, control_orders], bins=bins,
            label=[f'Epilepsy (n={len(epilepsy_orders):,})', 
                   f'Control (n={len(control_orders):,})'],
            alpha=0.7, edgecolor='black',
            color=['#e74c3c', '#3498db'],
            align='left')
    
    # Mark recommended order
    ax.axvline(recommended_order, color='red', linestyle='--', linewidth=2.5, 
               label=f'Recommended: p={recommended_order}')
    
    # Labels and title
    ax.set_xlabel('Optimal Order (p)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count (number of epochs)', fontsize=14, fontweight='bold')
    ax.set_title(f'Epoch-Level BIC Distribution by Group', 
                fontsize=16, fontweight='bold')
    
    ax.set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add group statistics
    stats_text = f"Epilepsy mean: {np.mean(epilepsy_orders):.2f}\n"
    stats_text += f"Control mean: {np.mean(control_orders):.2f}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('step1b_epoch_level_by_group.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: step1b_epoch_level_by_group.png")
    
    # =========================================================================
    # Figure 3: Complete hierarchy comparison (5 levels)
    # =========================================================================
    # Check if the other CSVs exist to create combined plot
    file_csv = Path('step1_file_level_orders.csv')
    session_csv = Path('step1_session_level_orders.csv')
    patient_csv = Path('step1_patient_level_orders.csv')
    
    if file_csv.exists() and session_csv.exists() and patient_csv.exists():
        print("\n📊 Creating complete 5-level hierarchy plot...")
        
        file_df = pd.read_csv(file_csv)
        session_df = pd.read_csv(session_csv)
        patient_df = pd.read_csv(patient_csv)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Epoch level (new!)
        axes[0, 0].hist(orders, bins=range(MIN_ORDER, MAX_ORDER + 2), 
                       edgecolor='black', alpha=0.7, color='steelblue', align='left')
        axes[0, 0].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Optimal Order (p)', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title(f'Epoch Level (n={len(orders):,})', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
        axes[0, 0].grid(True, alpha=0.3)
        
        # File level
        axes[0, 1].hist(file_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Mean Order', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title(f'File Level (n={len(file_df)})', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Session level
        axes[0, 2].hist(session_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 2].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
        axes[0, 2].set_xlabel('Mean Order', fontsize=11)
        axes[0, 2].set_ylabel('Count', fontsize=11)
        axes[0, 2].set_title(f'Session Level (n={len(session_df)})', fontsize=13, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Patient level
        axes[1, 0].hist(patient_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 0].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Mean Order', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title(f'Patient Level (n={len(patient_df)})', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Group comparison - epochs
        axes[1, 1].hist([epilepsy_orders, control_orders], bins=range(MIN_ORDER, MAX_ORDER + 2),
                       label=['Epilepsy', 'Control'], alpha=0.7, edgecolor='black',
                       color=['#e74c3c', '#3498db'], align='left')
        axes[1, 1].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Optimal Order (p)', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title('Group Comparison (Epochs)', fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
        axes[1, 1].grid(True, alpha=0.3)
        
        # Group comparison - patients
        epilepsy_patient = patient_df[patient_df['group'] == 'epilepsy']['mean_order']
        control_patient = patient_df[patient_df['group'] == 'control']['mean_order']
        
        axes[1, 2].hist([epilepsy_patient, control_patient], bins=15,
                       label=['Epilepsy', 'Control'], alpha=0.7, edgecolor='black',
                       color=['#e74c3c', '#3498db'])
        axes[1, 2].axvline(recommended_order, color='red', linestyle='--', linewidth=2,
                          label=f'Recommended: p={recommended_order}')
        axes[1, 2].set_xlabel('Mean Order', fontsize=11)
        axes[1, 2].set_ylabel('Count', fontsize=11)
        axes[1, 2].set_title('Group Comparison (Patients)', fontsize=13, fontweight='bold')
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('BIC-Selected Order Distribution: Complete Hierarchy (Epoch → File → Session → Patient → Group)', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig('step1b_complete_hierarchy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: step1b_complete_hierarchy.png")
    
    else:
        print("\n⚠️  Other CSV files not found - skipping combined hierarchy plot")
        print("    Run this script from the same directory as step1_*_orders.csv files")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 1B: Collect epoch-level BIC orders for histogram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis (all epochs)
  python step1b_epoch_level_histogram.py --input_dir data_pp_balanced
  
  # Sampled analysis (100 epochs per file - faster)
  python step1b_epoch_level_histogram.py --input_dir data_pp_balanced --sample_size 100
  
  # Quick test (10 epochs per file)
  python step1b_epoch_level_histogram.py --input_dir data_pp_balanced --sample_size 10
        """
    )
    
    parser.add_argument("--input_dir", required=True,
                       help="Root directory (e.g., data_pp_balanced)")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Epochs per file: None=ALL (default), N=sample N epochs")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 1)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("STEP 1B: EPOCH-LEVEL BIC HISTOGRAM")
    print("="*70)
    print()
    
    # Collect epoch-level orders
    df = collect_epoch_orders(
        args.input_dir,
        sample_size=args.sample_size,
        n_workers=args.workers
    )
    
    if df is None or len(df) == 0:
        print("❌ No data collected!")
        return
    
    # Save to CSV
    csv_path = 'step1b_epoch_level_orders.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("EPOCH-LEVEL STATISTICS")
    print("="*70)
    print(f"  Total epochs: {len(df):,}")
    print(f"  Mean order:   {df['optimal_order'].mean():.2f}")
    print(f"  Median order: {df['optimal_order'].median():.1f}")
    print(f"  Std order:    {df['optimal_order'].std():.2f}")
    print(f"  Mode order:   {df['optimal_order'].mode()[0]}")
    print(f"  Min order:    {df['optimal_order'].min()}")
    print(f"  Max order:    {df['optimal_order'].max()}")
    
    # Group breakdown
    print("\n  By Group:")
    for group in ['epilepsy', 'control']:
        group_data = df[df['group'] == group]['optimal_order']
        if len(group_data) > 0:
            print(f"    {group.capitalize():12} (n={len(group_data):,}): "
                  f"mean={group_data.mean():.2f}, median={group_data.median():.1f}")
    
    # Create visualizations
    create_epoch_histogram(df)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print("\nOutput files:")
    print("  - step1b_epoch_level_orders.csv")
    print("  - step1b_epoch_level_histogram.png")
    print("  - step1b_epoch_level_by_group.png")
    print("  - step1b_complete_hierarchy.png (if other CSVs available)")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()