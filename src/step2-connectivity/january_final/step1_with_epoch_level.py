"""
STEP 1: COMPREHENSIVE HIERARCHICAL BIC ANALYSIS (WITH EPOCH-LEVEL TRACKING)
============================================================================
Analyzes optimal model orders at ALL levels INCLUDING EPOCH LEVEL:
  - EPOCH level (each individual epoch) ← NEW!
  - File level (each .npy file)
  - Session level (each session folder)
  - Patient level (each patient folder)
  - Group level (epilepsy vs control)

FEATURES:
  - Saves EVERY epoch's optimal order (not just aggregates)
  - Parallel processing (uses all CPU cores)
  - Checkpointing (stop/resume anytime)

Usage:
    python step1_with_epoch_level.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced
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
import concurrent.futures
import multiprocessing
import json

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================
MIN_ORDER = 8
MAX_ORDER = 22

# ==============================================================================
# BIC COMPUTATION
# ==============================================================================

def compute_bic_for_epoch(data, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """Compute optimal BIC order for a single epoch."""
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


def process_file_with_epochs(args):
    """
    Process one file and return INDIVIDUAL EPOCH RESULTS.
    
    Returns list of dicts, one per epoch:
    [
        {'group': 'epilepsy', 'patient_id': 'xxx', ..., 'epoch_idx': 0, 'optimal_order': 13},
        {'group': 'epilepsy', 'patient_id': 'xxx', ..., 'epoch_idx': 1, 'optimal_order': 12},
        ...
    ]
    """
    f, sample_size, min_order, max_order = args
    
    # 1. Extract metadata from path
    try:
        path_parts = f.parts
        
        group = None
        patient_id = None
        session_id = None
        recording_id = None
        
        for i, part in enumerate(path_parts):
            if part == '00_epilepsy':
                group = 'epilepsy'
                patient_id = path_parts[i + 1]
                session_id = path_parts[i + 2]
                recording_id = path_parts[i + 3]
                break
            elif part == '01_no_epilepsy':
                group = 'control'
                patient_id = path_parts[i + 1]
                session_id = path_parts[i + 2]
                recording_id = path_parts[i + 3]
                break
        
        if group is None:
            return []
            
    except (IndexError, Exception):
        return []
    
    # 2. Load epochs
    try:
        epochs = np.load(f)
        n_epochs = len(epochs)
        
        # Sample if requested
        if sample_size is not None and sample_size < n_epochs:
            indices = np.random.choice(n_epochs, sample_size, replace=False)
        else:
            indices = np.arange(n_epochs)
        
    except Exception:
        return []
    
    # 3. Process each epoch and store INDIVIDUAL results
    epoch_results = []
    
    for idx in indices:
        order = compute_bic_for_epoch(epochs[idx], min_order, max_order)
        
        if order is not None:
            epoch_results.append({
                'group': group,
                'patient_id': patient_id,
                'session_id': session_id,
                'recording_id': recording_id,
                'file_name': f.name,
                'epoch_idx': int(idx),
                'optimal_order': int(order)
            })
    
    return epoch_results


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def comprehensive_bic_analysis_with_epochs(input_dir, sample_size=None, 
                                           min_order=MIN_ORDER, max_order=MAX_ORDER,
                                           checkpoint_dir='bic_checkpoints'):
    """
    Perform BIC analysis saving EPOCH-LEVEL data.
    """
    input_dir = Path(input_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_file = checkpoint_dir / 'processed_files.json'
    epoch_cache = checkpoint_dir / 'epoch_results_cache.csv'
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE BIC ANALYSIS (WITH EPOCH-LEVEL TRACKING)")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Sample size: {sample_size if sample_size else 'ALL EPOCHS'}")
    print(f"Order range: {min_order}-{max_order}")
    print(f"\n💾 CHECKPOINTING ENABLED - Press Ctrl+C to stop safely")
    print(f"{'='*80}\n")
    
    # Find all files
    all_files = list(input_dir.rglob("*_epochs.npy"))
    print(f"📁 Total files: {len(all_files)}\n")
    
    if len(all_files) == 0:
        print("❌ No epoch files found!")
        return None
    
    # Load checkpoint
    processed_files = set()
    all_epoch_results = []
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_files = set(checkpoint_data.get('processed_files', []))
            
            if epoch_cache.exists():
                cached_df = pd.read_csv(epoch_cache)
                all_epoch_results = cached_df.to_dict('records')
            
            print(f"✅ RESUMING FROM CHECKPOINT:")
            print(f"    Files done: {len(processed_files)}")
            print(f"    Epochs cached: {len(all_epoch_results):,}")
            print(f"    Remaining: {len(all_files) - len(processed_files)} files\n")
        except:
            processed_files = set()
            all_epoch_results = []
    
    # Filter files
    files_to_process = [f for f in all_files if str(f) not in processed_files]
    
    if len(files_to_process) == 0:
        print("✅ All files already processed!")
        epoch_df = pd.DataFrame(all_epoch_results)
    else:
        # Prepare tasks
        tasks = [(f, sample_size, min_order, max_order) for f in files_to_process]
        
        try:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = {executor.submit(process_file_with_epochs, task): task[0] for task in tasks}
                
                pbar = tqdm(total=len(files_to_process), desc="Processing", unit="file")
                
                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    
                    try:
                        epoch_results = future.result()
                        
                        if epoch_results:
                            all_epoch_results.extend(epoch_results)
                            processed_files.add(str(file_path))
                            
                            # Checkpoint every 10 files
                            if len(processed_files) % 10 == 0:
                                pd.DataFrame(all_epoch_results).to_csv(epoch_cache, index=False)
                                with open(checkpoint_file, 'w') as f:
                                    json.dump({
                                        'processed_files': list(processed_files),
                                        'n_epochs': len(all_epoch_results),
                                        'timestamp': pd.Timestamp.now().isoformat()
                                    }, f)
                            
                            pbar.set_postfix({'epochs': f"{len(all_epoch_results):,}"})
                    
                    except Exception as e:
                        print(f"\n⚠️ Error: {file_path.name}: {e}")
                    
                    pbar.update(1)
                
                pbar.close()
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ INTERRUPTED - Saving checkpoint...")
            pd.DataFrame(all_epoch_results).to_csv(epoch_cache, index=False)
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'processed_files': list(processed_files),
                    'n_epochs': len(all_epoch_results)
                }, f)
            print(f"✅ Saved {len(all_epoch_results):,} epochs. Run same command to resume.")
            return None
        
        # Final save
        epoch_df = pd.DataFrame(all_epoch_results)
        epoch_df.to_csv(epoch_cache, index=False)
    
    if len(epoch_df) == 0:
        print("❌ No valid results!")
        return None
    
    # ==========================================================================
    # SAVE EPOCH-LEVEL DATA (THE NEW PART!)
    # ==========================================================================
    
    print(f"\n{'='*80}")
    print(f"EPOCH-LEVEL RESULTS")
    print(f"{'='*80}")
    print(f"Total epochs analyzed: {len(epoch_df):,}")
    print(f"\nEpoch-level statistics:")
    print(f"  Mean order:   {epoch_df['optimal_order'].mean():.2f}")
    print(f"  Median order: {epoch_df['optimal_order'].median():.0f}")
    print(f"  Std order:    {epoch_df['optimal_order'].std():.2f}")
    print(f"  Mode order:   {epoch_df['optimal_order'].mode().iloc[0]}")
    
    # Save epoch-level CSV
    epoch_df.to_csv('step1_epoch_level_orders.csv', index=False)
    print(f"\n✅ Saved: step1_epoch_level_orders.csv ({len(epoch_df):,} rows)")
    
    # ==========================================================================
    # AGGREGATE TO HIGHER LEVELS
    # ==========================================================================
    
    # File level
    file_df = epoch_df.groupby(['group', 'patient_id', 'session_id', 'recording_id', 'file_name']).agg({
        'optimal_order': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).reset_index()
    file_df.columns = ['group', 'patient_id', 'session_id', 'recording_id', 'file_name',
                       'mean_order', 'median_order', 'std_order', 'min_order', 'max_order', 'n_epochs']
    file_df.to_csv('step1_file_level_orders.csv', index=False)
    print(f"✅ Saved: step1_file_level_orders.csv ({len(file_df)} files)")
    
    # Session level
    session_df = epoch_df.groupby(['group', 'patient_id', 'session_id']).agg({
        'optimal_order': ['mean', 'median', 'std', 'count']
    }).reset_index()
    session_df.columns = ['group', 'patient_id', 'session_id', 
                          'mean_order', 'median_order', 'std_order', 'n_epochs']
    session_df.to_csv('step1_session_level_orders.csv', index=False)
    print(f"✅ Saved: step1_session_level_orders.csv ({len(session_df)} sessions)")
    
    # Patient level
    patient_df = epoch_df.groupby(['group', 'patient_id']).agg({
        'optimal_order': ['mean', 'median', 'std', 'count']
    }).reset_index()
    patient_df.columns = ['group', 'patient_id', 'mean_order', 'median_order', 'std_order', 'n_epochs']
    patient_df.to_csv('step1_patient_level_orders.csv', index=False)
    print(f"✅ Saved: step1_patient_level_orders.csv ({len(patient_df)} patients)")
    
    # Group level
    group_df = epoch_df.groupby('group').agg({
        'optimal_order': ['mean', 'median', 'std', 'count']
    }).reset_index()
    group_df.columns = ['group', 'mean_order', 'median_order', 'std_order', 'n_epochs']
    group_df.to_csv('step1_group_level_orders.csv', index=False)
    print(f"✅ Saved: step1_group_level_orders.csv")
    
    # ==========================================================================
    # RECOMMENDATION
    # ==========================================================================
    
    global_mean = epoch_df['optimal_order'].mean()
    recommended_order = round(global_mean)
    
    print(f"\n{'='*80}")
    print(f"✅ RECOMMENDED FIXED ORDER: p = {recommended_order}")
    print(f"{'='*80}")
    print(f"Based on {len(epoch_df):,} individual epochs")
    print(f"Global mean: {global_mean:.2f}")
    
    # Group comparison
    epi_epochs = epoch_df[epoch_df['group'] == 'epilepsy']
    ctrl_epochs = epoch_df[epoch_df['group'] == 'control']
    
    print(f"\nGroup comparison (EPOCH LEVEL):")
    print(f"  Epilepsy: {len(epi_epochs):,} epochs, mean = {epi_epochs['optimal_order'].mean():.2f}")
    print(f"  Control:  {len(ctrl_epochs):,} epochs, mean = {ctrl_epochs['optimal_order'].mean():.2f}")
    
    # ==========================================================================
    # VISUALIZATIONS
    # ==========================================================================
    
    print(f"\n📊 Creating visualizations...")
    
    # Figure 1: Epoch-level histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # All epochs
    axes[0].hist(epoch_df['optimal_order'], bins=range(MIN_ORDER, MAX_ORDER + 2),
                 edgecolor='black', alpha=0.7, color='steelblue', align='left')
    axes[0].axvline(recommended_order, color='red', linestyle='--', linewidth=2,
                    label=f'Recommended: p={recommended_order}')
    axes[0].set_xlabel('Optimal Order (p)', fontsize=12)
    axes[0].set_ylabel('Count (epochs)', fontsize=12)
    axes[0].set_title(f'Epoch-Level BIC Distribution (n={len(epoch_df):,})', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f"Mean: {epoch_df['optimal_order'].mean():.2f}\n"
    stats_text += f"Median: {epoch_df['optimal_order'].median():.0f}\n"
    stats_text += f"Std: {epoch_df['optimal_order'].std():.2f}\n"
    stats_text += f"Mode: {epoch_df['optimal_order'].mode().iloc[0]}"
    axes[0].text(0.98, 0.98, stats_text, transform=axes[0].transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # By group
    axes[1].hist([epi_epochs['optimal_order'], ctrl_epochs['optimal_order']],
                 bins=range(MIN_ORDER, MAX_ORDER + 2),
                 label=[f'Epilepsy (n={len(epi_epochs):,})', f'Control (n={len(ctrl_epochs):,})'],
                 alpha=0.7, edgecolor='black', color=['#e74c3c', '#3498db'], align='left')
    axes[1].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Optimal Order (p)', fontsize=12)
    axes[1].set_ylabel('Count (epochs)', fontsize=12)
    axes[1].set_title('Epoch-Level BIC by Group', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step1_epoch_level_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: step1_epoch_level_histogram.png")
    
    # Figure 2: Complete hierarchy
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Epoch level
    axes[0, 0].hist(epoch_df['optimal_order'], bins=range(MIN_ORDER, MAX_ORDER + 2),
                    edgecolor='black', alpha=0.7, color='steelblue', align='left')
    axes[0, 0].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title(f'Epoch Level (n={len(epoch_df):,})', fontweight='bold')
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
    axes[1, 1].hist([epi_epochs['optimal_order'], ctrl_epochs['optimal_order']],
                    bins=range(MIN_ORDER, MAX_ORDER + 2),
                    label=['Epilepsy', 'Control'], alpha=0.7, edgecolor='black',
                    color=['#e74c3c', '#3498db'], align='left')
    axes[1, 1].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Group Comparison (Epochs)', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
    axes[1, 1].grid(True, alpha=0.3)
    
    # Group comparison - patients
    epi_pat = patient_df[patient_df['group'] == 'epilepsy']['mean_order']
    ctrl_pat = patient_df[patient_df['group'] == 'control']['mean_order']
    axes[1, 2].hist([epi_pat, ctrl_pat], bins=15,
                    label=['Epilepsy', 'Control'], alpha=0.7, edgecolor='black',
                    color=['#e74c3c', '#3498db'])
    axes[1, 2].axvline(recommended_order, color='red', linestyle='--', linewidth=2)
    axes[1, 2].set_title('Group Comparison (Patients)', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Complete BIC Hierarchy (Recommended p={recommended_order})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('step1_complete_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: step1_complete_hierarchy.png")
    
    # Cleanup checkpoints
    print(f"\n🗑️ Cleaning up checkpoints...")
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if epoch_cache.exists():
            epoch_cache.unlink()
        checkpoint_dir.rmdir()
        print(f"✅ Checkpoints removed")
    except:
        pass
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY OF OUTPUT FILES")
    print(f"{'='*80}")
    print(f"📊 step1_epoch_level_orders.csv    - {len(epoch_df):,} epochs (INDIVIDUAL)")
    print(f"📊 step1_file_level_orders.csv     - {len(file_df)} files")
    print(f"📊 step1_session_level_orders.csv  - {len(session_df)} sessions")
    print(f"📊 step1_patient_level_orders.csv  - {len(patient_df)} patients")
    print(f"📊 step1_group_level_orders.csv    - 2 groups")
    print(f"📊 step1_epoch_level_histogram.png")
    print(f"📊 step1_complete_hierarchy.png")
    print(f"{'='*80}")
    
    print(f"\n🎯 NEXT STEP:")
    print(f"  python step2_compute_connectivity.py \\")
    print(f"      --input_dir {input_dir} \\")
    print(f"      --output_dir connectivity_fixed_p{recommended_order} \\")
    print(f"      --fixed_order {recommended_order}")
    
    return recommended_order


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: BIC analysis with EPOCH-LEVEL tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis (ALL epochs)
  python step1_with_epoch_level.py --input_dir data_pp_balanced
  
  # Quick test (20 epochs per file)
  python step1_with_epoch_level.py --input_dir data_pp_balanced --sample_size 20
        """
    )
    
    parser.add_argument("--input_dir", required=True, help="Root directory with epochs")
    parser.add_argument("--sample_size", type=int, default=0,
                        help="Epochs per file: 0=ALL, N=sample N epochs")
    parser.add_argument("--min_order", type=int, default=MIN_ORDER)
    parser.add_argument("--max_order", type=int, default=MAX_ORDER)
    parser.add_argument("--checkpoint_dir", type=str, default='bic_checkpoints')
    
    args = parser.parse_args()
    
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    comprehensive_bic_analysis_with_epochs(
        args.input_dir,
        sample_size=sample_size,
        min_order=args.min_order,
        max_order=args.max_order,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()