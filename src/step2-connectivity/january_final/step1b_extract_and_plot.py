r"""
STEP 1B: EPOCH-LEVEL HISTOGRAM FROM PRE-COMPUTED ORDERS
========================================================
Reads .npz files that already contain optimal BIC orders per epoch
and creates histogram visualizations.

MUCH FASTER than recomputing BIC - just reads existing data!

Usage:
    cd C:\Users\georg
    
    python step1b_extract_and_plot.py --input_dir "F:\path\to\connectivity\output"
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONSTANTS
# ==============================================================================
MIN_ORDER = 5
MAX_ORDER = 18
RECOMMENDED_ORDER = 13  # From your main analysis


# ==============================================================================
# EXTRACTION (Reads .npz files)
# ==============================================================================

def process_npz_file(file_path):
    """
    Opens a single .npz file and extracts metadata + orders.
    """
    try:
        data = np.load(file_path)
        
        # Check if 'orders' exists
        if 'orders' not in data:
            return []
        
        orders = data['orders']
        indices = data['indices'] if 'indices' in data else range(len(orders))
        
        # --- Metadata Extraction ---
        parts = file_path.parts
        
        group = "unknown"
        patient_id = "unknown"
        session_id = "unknown"
        recording_id = "unknown"
        
        if '00_epilepsy' in parts:
            group = 'epilepsy'
            idx = parts.index('00_epilepsy')
        elif '01_no_epilepsy' in parts:
            group = 'control'
            idx = parts.index('01_no_epilepsy')
        else:
            idx = -1

        if idx != -1 and len(parts) > idx + 3:
            patient_id = parts[idx + 1]
            session_id = parts[idx + 2]
            recording_id = parts[idx + 3]
        else:
            # Fallback: Parse from filename
            stem_parts = file_path.stem.split('_')
            if len(stem_parts) >= 3:
                patient_id = stem_parts[0]
                session_id = stem_parts[1]
                recording_id = file_path.parent.name

        # Create rows (one per epoch)
        results = []
        for i, order in enumerate(orders):
            results.append({
                'group': group,
                'patient_id': patient_id,
                'session_id': session_id,
                'recording_id': recording_id,
                'file_name': file_path.name,
                'epoch_idx': int(indices[i]) if hasattr(indices, '__getitem__') else i,
                'optimal_order': int(order)
            })
            
        return results

    except Exception as e:
        return []


def extract_all_orders(input_dir, n_workers=8):
    """
    Extract orders from all .npz files.
    """
    input_path = Path(input_dir)
    
    # Find all .npz files
    print(f"🔍 Scanning {input_path}...")
    npz_files = list(input_path.rglob("*_graphs.npz"))
    
    if not npz_files:
        # Try alternative pattern
        npz_files = list(input_path.rglob("*.npz"))
    
    if not npz_files:
        print("❌ No .npz files found!")
        return None

    print(f"📂 Found {len(npz_files)} files")
    print(f"🔧 Using {n_workers} parallel workers")
    print()

    # Process in parallel
    all_data = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_npz_file, f): f for f in npz_files}
        
        for future in tqdm(concurrent.futures.as_completed(futures), 
                           total=len(npz_files), desc="Extracting orders"):
            result = future.result()
            if result:
                all_data.extend(result)
    
    if not all_data:
        print("❌ No data extracted!")
        return None
    
    df = pd.DataFrame(all_data)
    print(f"\n✅ Extracted {len(df):,} epoch-level orders")
    
    return df


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def create_epoch_histogram(df, recommended_order=RECOMMENDED_ORDER):
    """
    Create epoch-level histogram visualization.
    """
    print("\n📊 Creating visualizations...")
    
    orders = df['optimal_order'].values
    
    # =========================================================================
    # Figure 1: Simple epoch-level histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    counts, bins, patches = ax.hist(
        orders, 
        bins=range(MIN_ORDER, MAX_ORDER + 2),
        edgecolor='black', 
        alpha=0.7, 
        color='steelblue',
        align='left'
    )
    
    ax.axvline(recommended_order, color='red', linestyle='--', linewidth=2.5, 
               label=f'Recommended: p={recommended_order}')
    
    # Add count labels on bars
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width()/2, count + max(counts)*0.01,
                    f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Optimal Order (p)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count (number of epochs)', fontsize=14, fontweight='bold')
    ax.set_title(f'Epoch-Level BIC Distribution (n={len(orders):,} epochs)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
    
    # Statistics box
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
    
    ax.axvline(recommended_order, color='red', linestyle='--', linewidth=2.5, 
               label=f'Recommended: p={recommended_order}')
    
    ax.set_xlabel('Optimal Order (p)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count (number of epochs)', fontsize=14, fontweight='bold')
    ax.set_title(f'Epoch-Level BIC Distribution by Group', fontsize=16, fontweight='bold')
    ax.set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add group statistics
    stats_text = ""
    if len(epilepsy_orders) > 0:
        stats_text += f"Epilepsy mean: {np.mean(epilepsy_orders):.2f}\n"
    if len(control_orders) > 0:
        stats_text += f"Control mean: {np.mean(control_orders):.2f}"
    
    if stats_text:
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('step1b_epoch_level_by_group.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: step1b_epoch_level_by_group.png")
    
    # =========================================================================
    # Figure 3: Complete 5-level hierarchy
    # =========================================================================
    file_csv = Path('step1_file_level_orders.csv')
    session_csv = Path('step1_session_level_orders.csv')
    patient_csv = Path('step1_patient_level_orders.csv')
    
    if file_csv.exists() and session_csv.exists() and patient_csv.exists():
        print("\n📊 Creating complete 5-level hierarchy plot...")
        
        try:
            file_df = pd.read_csv(file_csv)
            session_df = pd.read_csv(session_csv)
            patient_df = pd.read_csv(patient_csv)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            # Epoch level
            axes[0, 0].hist(orders, bins=range(MIN_ORDER, MAX_ORDER + 2), 
                            edgecolor='black', alpha=0.7, color='steelblue', align='left')
            axes[0, 0].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
            axes[0, 0].set_title(f'Epoch Level (n={len(orders):,})', fontweight='bold')
            axes[0, 0].set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
            axes[0, 0].grid(True, alpha=0.3)
            
            # File level
            axes[0, 1].hist(file_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 1].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
            axes[0, 1].set_title(f'File Level (n={len(file_df)})', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Session level
            axes[0, 2].hist(session_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            axes[0, 2].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
            axes[0, 2].set_title(f'Session Level (n={len(session_df)})', fontweight='bold')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Patient level
            axes[1, 0].hist(patient_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            axes[1, 0].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_title(f'Patient Level (n={len(patient_df)})', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Group comparison - epochs
            axes[1, 1].hist([epilepsy_orders, control_orders], bins=range(MIN_ORDER, MAX_ORDER + 2),
                            label=['Epilepsy', 'Control'], alpha=0.7, edgecolor='black',
                            color=['#e74c3c', '#3498db'], align='left')
            axes[1, 1].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
            axes[1, 1].set_title('Group Comparison (Epochs)', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
            axes[1, 1].grid(True, alpha=0.3)
            
            # Group comparison - patients
            ep_pat = patient_df[patient_df['group'] == 'epilepsy']['mean_order']
            ct_pat = patient_df[patient_df['group'] == 'control']['mean_order']
            
            axes[1, 2].hist([ep_pat, ct_pat], bins=15,
                            label=['Epilepsy', 'Control'], alpha=0.7, edgecolor='black',
                            color=['#e74c3c', '#3498db'])
            axes[1, 2].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
            axes[1, 2].set_title('Group Comparison (Patients)', fontweight='bold')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.suptitle('BIC-Selected Order Hierarchy', fontsize=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig('step1b_complete_hierarchy.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved: step1b_complete_hierarchy.png")
            
        except Exception as e:
            print(f"\n⚠️ Could not create hierarchy plot: {e}")
    else:
        print("\n⚠️ Other CSV files not found - skipping combined hierarchy plot")
        print("   (Ensure step1_file_level_orders.csv etc. are in this folder)")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract pre-computed BIC orders from .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing .npz files with pre-computed orders")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("STEP 1B: EPOCH-LEVEL HISTOGRAM (FROM PRE-COMPUTED ORDERS)")
    print("="*70)
    print()
    
    # Extract orders from .npz files
    df = extract_all_orders(args.input_dir, n_workers=args.workers)
    
    if df is None or len(df) == 0:
        print("❌ No data extracted!")
        return
    
    # Save CSV
    csv_path = 'step1b_epoch_level_orders.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved: {csv_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("EPOCH-LEVEL STATISTICS")
    print("="*70)
    print(f"  Total epochs: {len(df):,}")
    print(f"  Mean order:   {df['optimal_order'].mean():.2f}")
    print(f"  Median order: {df['optimal_order'].median():.1f}")
    print(f"  Std order:    {df['optimal_order'].std():.2f}")
    print(f"  Mode order:   {df['optimal_order'].mode()[0]}")
    
    # Create visualizations
    create_epoch_histogram(df)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()