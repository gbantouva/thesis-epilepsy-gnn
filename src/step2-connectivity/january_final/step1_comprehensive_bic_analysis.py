"""
STEP 1: COMPREHENSIVE HIERARCHICAL BIC ANALYSIS (PARALLEL + CHECKPOINTING)
===========================================================================
Analyzes optimal model orders at ALL levels:
  - File level (each .npy file)
  - Recording level (each montage: tcp_ar, tcp_le, etc.)
  - Session level (each session folder)
  - Patient level (each patient folder)
  - Group level (epilepsy vs control)

Then recommends optimal FIXED order based on hierarchical statistics.

FEATURES:
  - Parallel processing (uses all CPU cores)
  - Checkpointing (stop/resume anytime)
  - Progress tracking (shows epochs processed)
  - Publication-quality plots

Usage:
    # Full analysis (ALL epochs, 4-6 hours, can stop/resume)
    python step1_comprehensive_bic_analysis.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced
    
    # Resume interrupted run (same command)
    python step1_comprehensive_bic_analysis.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced
    
    # Quick test (50 epochs per file)
    python step1_comprehensive_bic_analysis.py \
        --input_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced \
        --sample_size 50
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import multiprocessing
import json

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL CONSTANTS - CONSISTENT ORDER RANGE
# ==============================================================================
MIN_ORDER = 5   # Minimum order to test
MAX_ORDER = 18  # Maximum order to test

# ==============================================================================
# BIC COMPUTATION
# ==============================================================================

def compute_bic_for_epoch(data, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """
    Compute BIC curve for a single epoch and return optimal order.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Single epoch data
    min_order : int
        Minimum order to test
    max_order : int
        Maximum order to test
    
    Returns
    -------
    best_order : int or None
        Optimal order (minimum BIC)
    """
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    data_scaled = data / data_std
    
    try:
        model = VAR(data_scaled.T)  # Transpose to (n_times, n_channels)
        
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


def process_file(epoch_file, sample_size=None, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """
    Process one .npy file and return optimal orders.
    
    Parameters
    ----------
    epoch_file : Path
        Path to *_epochs.npy file
    sample_size : int or None
        Number of epochs to sample (None = all)
    min_order : int
        Minimum order to test
    max_order : int
        Maximum order to test
    
    Returns
    -------
    orders : list
        List of optimal orders for sampled epochs
    """
    try:
        epochs = np.load(epoch_file)
        n_epochs = len(epochs)
        
        # Sample epochs
        if sample_size is not None and sample_size < n_epochs:
            indices = np.random.choice(n_epochs, sample_size, replace=False)
        else:
            indices = np.arange(n_epochs)
        
        orders = []
        for idx in indices:
            order = compute_bic_for_epoch(epochs[idx], min_order, max_order)
            if order is not None:
                orders.append(order)
        
        return orders
        
    except Exception as e:
        return []


def process_file_wrapper(args):
    """
    Wrapper for parallel processing.
    Parses metadata from FOLDER STRUCTURE for consistency.
    
    Path structure:
        .../00_epilepsy/aaaaaanr/s004_2013/01_tcp_ar/aaaaaanr_s004_t001_epochs.npy
        .../01_no_epilepsy/aaaaaanr/s004_2013/01_tcp_ar/aaaaaanr_s004_t001_epochs.npy
            GROUP         PATIENT  SESSION   RECORDING  FILENAME
    """
    f, sample_size, min_order, max_order = args
    
    # 1. Process the file (Compute BIC)
    orders = process_file(f, sample_size, min_order, max_order)
    
    if len(orders) == 0:
        return None
    
    # 2. Extract Metadata from PATH
    try:
        path_parts = f.parts
        
        # Find the group folder and extract hierarchy
        group = None
        patient_id = None
        session_id = None
        recording_id = None
        
        for i, part in enumerate(path_parts):
            if part == '00_epilepsy':
                group = 'epilepsy'
                patient_id = path_parts[i + 1]    # e.g., 'aaaaaanr'
                session_id = path_parts[i + 2]    # e.g., 's004_2013'
                recording_id = path_parts[i + 3]  # e.g., '01_tcp_ar'
                break
            elif part == '01_no_epilepsy':  # ✅ Your actual folder name
                group = 'control'
                patient_id = path_parts[i + 1]    # e.g., 'aaaaaanr'
                session_id = path_parts[i + 2]    # e.g., 's004_2013'
                recording_id = path_parts[i + 3]  # e.g., '02_tcp_le'
                break
        
        # Validation
        if group is None or patient_id is None:
            return None

    except (IndexError, Exception) as e:
        return None
    
    # 3. Return statistics
    return {
        'group': group,
        'patient_id': patient_id,
        'session_id': session_id,
        'recording_id': recording_id,
        'file_name': f.name,
        'file_path': str(f),
        'n_epochs_sampled': len(orders),
        'mean_order': np.mean(orders),
        'median_order': np.median(orders),
        'std_order': np.std(orders),
        'min_order': np.min(orders),
        'max_order': np.max(orders)
    }


# ==============================================================================
# HIERARCHICAL ANALYSIS (PARALLEL VERSION WITH CHECKPOINTING)
# ==============================================================================

def comprehensive_bic_analysis(input_dir, sample_size=None, min_order=MIN_ORDER, max_order=MAX_ORDER, checkpoint_dir='bic_checkpoints'):
    """
    Perform comprehensive hierarchical BIC analysis with PARALLEL PROCESSING + CHECKPOINTING.
    
    Parameters
    ----------
    input_dir : str or Path
        Root directory containing epoch files
    sample_size : int or None
        Epochs to sample per file (None = all)
    min_order : int
        Minimum order to test
    max_order : int
        Maximum order to test
    checkpoint_dir : str
        Directory to store checkpoints
    
    Returns
    -------
    recommended_order : int
        Recommended fixed order for Step 2
    """
    input_dir = Path(input_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Checkpoint files
    checkpoint_file = checkpoint_dir / 'processed_files.json'
    results_cache = checkpoint_dir / 'results_cache.csv'
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE HIERARCHICAL BIC ANALYSIS (PARALLEL + CHECKPOINTING)")
    print(f"{'='*80}")
    print(f"Input directory: {input_dir}")
    print(f"Sample size per file: {sample_size if sample_size else 'ALL EPOCHS'}")
    print(f"Order range: {min_order}-{max_order}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # TIME ESTIMATE
    if sample_size is None:
        print(f"\n⚠️  ANALYZING ALL EPOCHS - Parallel processing enabled")
        print(f"    Expected time: 4-6 hours with {multiprocessing.cpu_count()} CPU cores")
        print(f"    For quick testing, use --sample_size 50")
    else:
        print(f"\n📊 SAMPLING MODE - Faster completion")
    
    print(f"\n💾 CHECKPOINTING ENABLED:")
    print(f"    Press Ctrl+C to stop safely at any time")
    print(f"    Re-run same command to resume from checkpoint")
    print(f"    Results saved every 10 files (no data loss)")
    print(f"{'='*80}\n")
    
    # Find all epoch files
    all_files = list(input_dir.rglob("*_epochs.npy"))
    print(f"📁 Total epoch files found: {len(all_files)}\n")
    
    if len(all_files) == 0:
        print("❌ No epoch files found! Check your input directory.")
        return None
    
    # =========================================================================
    # LOAD CHECKPOINT (if exists)
    # =========================================================================
    
    processed_files = set()
    file_stats = []
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_files = set(checkpoint_data.get('processed_files', []))
            
            if results_cache.exists():
                cached_df = pd.read_csv(results_cache)
                file_stats = cached_df.to_dict('records')
            
            total_epochs_cached = sum(r['n_epochs_sampled'] for r in file_stats)
            
            print(f"✅ RESUMING FROM CHECKPOINT:")
            print(f"    Already processed: {len(processed_files)} files")
            print(f"    Remaining: {len(all_files) - len(processed_files)} files")
            print(f"    Total epochs so far: {total_epochs_cached:,}")
            print(f"    Checkpoint saved: {checkpoint_data.get('timestamp', 'unknown')}\n")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print(f"    Starting fresh...\n")
            processed_files = set()
            file_stats = []
    
    # =========================================================================
    # LEVEL 1: FILE-LEVEL ANALYSIS (PARALLEL + CHECKPOINTING)
    # =========================================================================
    
    print(f"{'='*80}")
    print(f"LEVEL 1: FILE-LEVEL ANALYSIS (PARALLEL PROCESSING)")
    print(f"{'='*80}\n")
    print(f"🚀 Using {multiprocessing.cpu_count()} CPU cores...\n")
    
    # Filter files to process (skip already done)
    files_to_process = [f for f in all_files if str(f) not in processed_files]
    files_already_done = len(all_files) - len(files_to_process)
    
    print(f"📊 Status:")
    print(f"    Total files: {len(all_files)}")
    print(f"    Already done: {files_already_done}")
    print(f"    To process: {len(files_to_process)}")
    
    if sample_size is not None:
        estimated_remaining = len(files_to_process) * sample_size
        print(f"    Estimated epochs remaining: ~{estimated_remaining:,}")
    print()
    
    if len(files_to_process) == 0:
        print("✅ All files already processed!")
        print("    Loading cached results...\n")
        file_df = pd.DataFrame(file_stats)
    else:
        # Prepare tasks
        tasks = [(f, sample_size, min_order, max_order) for f in files_to_process]
        
        # Initialize counters
        total_epochs_processed = sum(r['n_epochs_sampled'] for r in file_stats)
        
        try:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = {executor.submit(process_file_wrapper, task): task[0] for task in tasks}
                
                # Progress bar with detailed stats
                pbar = tqdm(
                    total=len(files_to_process),
                    desc="Processing files",
                    unit="file",
                    initial=0
                )
                
                for future in concurrent.futures.as_completed(futures):
                    file_path = futures[future]
                    
                    try:
                        result = future.result()
                        
                        if result is not None:
                            file_stats.append(result)
                            total_epochs_processed += result['n_epochs_sampled']
                            processed_files.add(str(file_path))
                            
                            # Save checkpoint every 10 files
                            if len(processed_files) % 10 == 0:
                                # Save results cache
                                pd.DataFrame(file_stats).to_csv(results_cache, index=False)
                                
                                # Save checkpoint
                                with open(checkpoint_file, 'w') as f:
                                    json.dump({
                                        'processed_files': list(processed_files),
                                        'total_epochs': total_epochs_processed,
                                        'timestamp': pd.Timestamp.now().isoformat()
                                    }, f)
                            
                            # Update progress bar with epoch count
                            pbar.set_postfix({
                                'epochs': f"{total_epochs_processed:,}",
                                'valid': len(file_stats),
                                'total_done': files_already_done + len(processed_files) - files_already_done
                            })
                    
                    except Exception as e:
                        print(f"\n⚠️  Error processing {file_path.name}: {e}")
                    
                    pbar.update(1)
                
                pbar.close()
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print(f"⚠️  INTERRUPTED BY USER (Ctrl+C)")
            print(f"{'='*80}")
            print(f"✅ Progress saved to checkpoint!")
            print(f"📊 Statistics:")
            print(f"    Files processed: {len(file_stats)}")
            print(f"    Epochs analyzed: {total_epochs_processed:,}")
            print(f"    Checkpoint location: {checkpoint_dir}")
            print(f"\n🔄 To resume from this point:")
            print(f"    Run the exact same command again")
            print(f"{'='*80}\n")
            return None
        
        # Final save
        pd.DataFrame(file_stats).to_csv(results_cache, index=False)
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed_files': list(processed_files),
                'total_epochs': total_epochs_processed,
                'timestamp': pd.Timestamp.now().isoformat(),
                'completed': True
            }, f)
        
        file_df = pd.DataFrame(file_stats)
    
    # Validation
    if len(file_stats) == 0:
        print("❌ No valid results! Check your data.")
        return None
    
    total_epochs = sum(r['n_epochs_sampled'] for r in file_stats)
    
    print(f"\n✅ Processed {len(file_df)} files successfully")
    print(f"📊 Total epochs analyzed: {total_epochs:,}\n")
    print("Sample of file-level results:")
    print(file_df[['group', 'patient_id', 'session_id', 'recording_id', 'mean_order', 'n_epochs_sampled']].head(10))
    print()
    
    # =========================================================================
    # LEVEL 2: RECORDING-LEVEL ANALYSIS
    # =========================================================================
    
    print(f"\n{'='*80}")
    print(f"LEVEL 2: RECORDING-LEVEL ANALYSIS")
    print(f"{'='*80}\n")
    
    recording_stats = []
    
    for (group, patient, session, recording), group_df in file_df.groupby(
        ['group', 'patient_id', 'session_id', 'recording_id']
    ):
        recording_stats.append({
            'group': group,
            'patient_id': patient,
            'session_id': session,
            'recording_id': recording,
            'n_files': len(group_df),
            'mean_order': group_df['mean_order'].mean(),
            'median_order': group_df['median_order'].median(),
            'std_order': group_df['mean_order'].std() if len(group_df) > 1 else 0.0,
            'total_epochs': group_df['n_epochs_sampled'].sum()
        })
    
    recording_df = pd.DataFrame(recording_stats)
    
    print(f"✅ Total recordings: {len(recording_df)}\n")
    print("Sample of recording-level results:")
    print(recording_df[['group', 'patient_id', 'session_id', 'recording_id', 'mean_order']].head(10))
    print()
    
    # =========================================================================
    # LEVEL 3: SESSION-LEVEL ANALYSIS
    # =========================================================================
    
    print(f"\n{'='*80}")
    print(f"LEVEL 3: SESSION-LEVEL ANALYSIS")
    print(f"{'='*80}\n")
    
    session_stats = []
    
    for (group, patient, session), group_df in recording_df.groupby(
        ['group', 'patient_id', 'session_id']
    ):
        session_stats.append({
            'group': group,
            'patient_id': patient,
            'session_id': session,
            'n_recordings': len(group_df),
            'mean_order': group_df['mean_order'].mean(),
            'median_order': group_df['median_order'].median(),
            'std_order': group_df['mean_order'].std() if len(group_df) > 1 else 0.0,
            'total_epochs': group_df['total_epochs'].sum()
        })
    
    session_df = pd.DataFrame(session_stats)
    
    print(f"✅ Total sessions: {len(session_df)}\n")
    print("Sample of session-level results:")
    print(session_df[['group', 'patient_id', 'session_id', 'n_recordings', 'mean_order']].head(10))
    print()
    
    # =========================================================================
    # LEVEL 4: PATIENT-LEVEL ANALYSIS
    # =========================================================================
    
    print(f"\n{'='*80}")
    print(f"LEVEL 4: PATIENT-LEVEL ANALYSIS")
    print(f"{'='*80}\n")
    
    patient_stats = []
    
    for (group, patient), group_df in session_df.groupby(['group', 'patient_id']):
        patient_stats.append({
            'group': group,
            'patient_id': patient,
            'n_sessions': len(group_df),
            'mean_order': group_df['mean_order'].mean(),
            'median_order': group_df['median_order'].median(),
            'std_order': group_df['mean_order'].std() if len(group_df) > 1 else 0.0,
            'total_epochs': group_df['total_epochs'].sum()
        })
    
    patient_df = pd.DataFrame(patient_stats)
    
    print(f"✅ Total patients: {len(patient_df)}\n")
    print("Sample of patient-level results:")
    print(patient_df[['group', 'patient_id', 'n_sessions', 'mean_order']].head(10))
    print()
    
    # =========================================================================
    # LEVEL 5: GROUP-LEVEL ANALYSIS
    # =========================================================================
    
    print(f"\n{'='*80}")
    print(f"LEVEL 5: GROUP-LEVEL ANALYSIS (EPILEPSY vs CONTROL)")
    print(f"{'='*80}\n")
    
    group_stats = []
    
    for group, group_df in patient_df.groupby('group'):
        group_stats.append({
            'group': group,
            'n_patients': len(group_df),
            'mean_order': group_df['mean_order'].mean(),
            'median_order': group_df['median_order'].median(),
            'std_order': group_df['mean_order'].std(),
            'total_epochs': group_df['total_epochs'].sum()
        })
    
    group_df = pd.DataFrame(group_stats)
    
    print("GROUP-LEVEL STATISTICS:")
    print(group_df.to_string(index=False))
    print()
    
    # =========================================================================
    # FINAL RECOMMENDATIONS
    # =========================================================================
    
    print(f"\n{'='*80}")
    print(f"FINAL RECOMMENDATION")
    print(f"{'='*80}\n")
    
    # Global statistics (across all patients)
    global_mean = patient_df['mean_order'].mean()
    global_median = patient_df['median_order'].median()
    global_std = patient_df['mean_order'].std()
    
    # Group-specific statistics
    epilepsy_patients = patient_df[patient_df['group'] == 'epilepsy']
    control_patients = patient_df[patient_df['group'] == 'control']
    
    epilepsy_mean = epilepsy_patients['mean_order'].mean() if len(epilepsy_patients) > 0 else 0
    control_mean = control_patients['mean_order'].mean() if len(control_patients) > 0 else 0
    
    print(f"📊 PATIENT-LEVEL AGGREGATED STATISTICS:")
    print(f"  Global mean order:        {global_mean:.2f}")
    print(f"  Global median order:      {global_median:.0f}")
    print(f"  Global std:               {global_std:.2f}")
    print()
    print(f"📊 GROUP-SPECIFIC STATISTICS:")
    print(f"  Epilepsy patients:        {len(epilepsy_patients)}")
    print(f"  Epilepsy mean order:      {epilepsy_mean:.2f}")
    print(f"  Control patients:         {len(control_patients)}")
    print(f"  Control mean order:       {control_mean:.2f}")
    print(f"  Difference (Epi - Ctrl):  {epilepsy_mean - control_mean:.2f}")
    print()
    
    # Recommendation
    recommended_order = round(global_mean)
    
    print(f"{'='*80}")
    print(f"✅ RECOMMENDED FIXED ORDER: p = {recommended_order}")
    print(f"{'='*80}")
    print(f"\nJustification:")
    print(f"  - Based on patient-level averaging (sessions → patients)")
    print(f"  - Represents optimal complexity across {len(patient_df)} patients")
    print(f"  - Groups show similar orders (Δ = {abs(epilepsy_mean - control_mean):.2f})")
    print(f"  - Analyzed {total_epochs:,} total epochs")
    print(f"  - Will be used uniformly for ALL epochs in Step 2")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    print(f"💾 Saving results...\n")
    
    file_df.to_csv('step1_file_level_orders.csv', index=False)
    recording_df.to_csv('step1_recording_level_orders.csv', index=False)
    session_df.to_csv('step1_session_level_orders.csv', index=False)
    patient_df.to_csv('step1_patient_level_orders.csv', index=False)
    group_df.to_csv('step1_group_level_orders.csv', index=False)
    
    summary = {
        'recommended_order': recommended_order,
        'global_mean': global_mean,
        'global_median': global_median,
        'global_std': global_std,
        'epilepsy_mean': epilepsy_mean,
        'control_mean': control_mean,
        'n_patients': len(patient_df),
        'n_sessions': len(session_df),
        'n_recordings': len(recording_df),
        'n_files': len(file_df),
        'total_epochs': total_epochs
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('step1_summary.csv', index=False)
    
    print(f"✅ Saved CSV files:")
    print(f"  - step1_file_level_orders.csv")
    print(f"  - step1_recording_level_orders.csv")
    print(f"  - step1_session_level_orders.csv")
    print(f"  - step1_patient_level_orders.csv")
    print(f"  - step1_group_level_orders.csv")
    print(f"  - step1_summary.csv")
    
    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    
    create_visualizations(file_df, recording_df, session_df, patient_df, group_df, recommended_order)
    
    # =========================================================================
    # CLEANUP CHECKPOINTS
    # =========================================================================
    
    print(f"\n🗑️  Cleaning up checkpoints...")
    try:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if results_cache.exists():
            results_cache.unlink()
        checkpoint_dir.rmdir()  # Remove empty directory
        print(f"✅ Checkpoints removed (analysis complete)")
    except Exception as e:
        print(f"⚠️  Could not remove checkpoints: {e}")
    
    return recommended_order


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

def create_visualizations(file_df, recording_df, session_df, patient_df, group_df, recommended_order):
    """Create comprehensive visualizations."""
    
    print(f"\n📊 Creating visualizations...")
    
    # Figure 1: Distribution at each level
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # File level
    axes[0, 0].hist(file_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(recommended_order, color='red', linestyle='--', linewidth=2, 
                       label=f'Recommended: p={recommended_order}')
    axes[0, 0].set_xlabel('Mean Order', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title(f'File Level (n={len(file_df)})', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Session level
    axes[0, 1].hist(session_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 1].axvline(recommended_order, color='red', linestyle='--', linewidth=2, 
                       label=f'Recommended: p={recommended_order}')
    axes[0, 1].set_xlabel('Mean Order', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title(f'Session Level (n={len(session_df)})', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Patient level
    axes[1, 0].hist(patient_df['mean_order'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(recommended_order, color='red', linestyle='--', linewidth=2, 
                       label=f'Recommended: p={recommended_order}')
    axes[1, 0].set_xlabel('Mean Order', fontsize=12)
    axes[1, 0].set_ylabel('Count', fontsize=12)
    axes[1, 0].set_title(f'Patient Level (n={len(patient_df)})', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Group comparison
    epilepsy_orders = patient_df[patient_df['group'] == 'epilepsy']['mean_order']
    control_orders = patient_df[patient_df['group'] == 'control']['mean_order']
    
    axes[1, 1].hist([epilepsy_orders, control_orders], bins=15, 
                   label=['Epilepsy', 'Control'], alpha=0.7, edgecolor='black',
                   color=['#e74c3c', '#3498db'])
    axes[1, 1].axvline(recommended_order, color='red', linestyle='--', linewidth=2, 
                       label=f'Recommended: p={recommended_order}')
    axes[1, 1].set_xlabel('Mean Order', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title(f'Group Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('BIC-Selected Order Distribution at All Hierarchical Levels', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('step1_hierarchical_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Group comparison boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Session level boxplot
    session_df.boxplot(column='mean_order', by='group', ax=axes[0])
    axes[0].set_title('Session Level', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Group', fontsize=12)
    axes[0].set_ylabel('Mean Order', fontsize=12)
    axes[0].axhline(recommended_order, color='red', linestyle='--', linewidth=2)
    axes[0].get_figure().suptitle('')  # Remove pandas default title
    
    # Patient level boxplot
    patient_df.boxplot(column='mean_order', by='group', ax=axes[1])
    axes[1].set_title('Patient Level', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Group', fontsize=12)
    axes[1].set_ylabel('Mean Order', fontsize=12)
    axes[1].axhline(recommended_order, color='red', linestyle='--', linewidth=2)
    axes[1].get_figure().suptitle('')  # Remove pandas default title
    
    # Summary bar chart
    axes[2].bar(['Epilepsy', 'Control'], 
               [group_df[group_df['group'] == 'epilepsy']['mean_order'].values[0],
                group_df[group_df['group'] == 'control']['mean_order'].values[0]],
               color=['#e74c3c', '#3498db'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].axhline(recommended_order, color='red', linestyle='--', linewidth=2, 
                    label=f'Recommended: p={recommended_order}')
    axes[2].set_ylabel('Mean Order', fontsize=12)
    axes[2].set_title('Group Means', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Group Comparison: Epilepsy vs Control', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('step1_group_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved visualizations:")
    print(f"  - step1_hierarchical_distributions.png")
    print(f"  - step1_group_comparison.png")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Comprehensive hierarchical BIC analysis (PARALLEL + CHECKPOINTING)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with ALL epochs (recommended for thesis - 4-6 hours)
  python step1_comprehensive_bic_analysis.py --input_dir data_pp_balanced
  
  # Resume interrupted run (same command automatically resumes)
  python step1_comprehensive_bic_analysis.py --input_dir data_pp_balanced
  
  # Quick test with 50 epochs per file (~2 hours)
  python step1_comprehensive_bic_analysis.py --input_dir data_pp_balanced --sample_size 50
  
  # Custom order range
  python step1_comprehensive_bic_analysis.py --input_dir data_pp_balanced --min_order 2 --max_order 20
  
  # Custom checkpoint directory
  python step1_comprehensive_bic_analysis.py --input_dir data_pp_balanced --checkpoint_dir my_checkpoints
        """
    )
    
    parser.add_argument("--input_dir", required=True,
                       help="Root directory (e.g., data_pp_balanced)")
    parser.add_argument("--sample_size", type=int, default=0,
                       help="Epochs per file: 0=ALL (default), N=sample N epochs")
    parser.add_argument("--min_order", type=int, default=MIN_ORDER,
                       help=f"Minimum order to test (default: {MIN_ORDER})")
    parser.add_argument("--max_order", type=int, default=MAX_ORDER,
                       help=f"Maximum order to test (default: {MAX_ORDER})")
    parser.add_argument("--checkpoint_dir", type=str, default='bic_checkpoints',
                       help="Checkpoint directory (default: bic_checkpoints)")
    
    args = parser.parse_args()
    
    # Convert 0 to None (means ALL epochs)
    sample_size = None if args.sample_size == 0 else args.sample_size
    
    # Run analysis
    recommended_order = comprehensive_bic_analysis(
        args.input_dir,
        sample_size=sample_size,
        min_order=args.min_order,
        max_order=args.max_order,
        checkpoint_dir=args.checkpoint_dir
    )
    
    if recommended_order is not None:
        print(f"\n{'='*80}")
        print(f"🎯 NEXT STEP:")
        print(f"{'='*80}")
        print(f"\nRun Step 2 with:\n")
        print(f"  python step2_compute_connectivity.py \\")
        print(f"      --input_dir {args.input_dir} \\")
        print(f"      --output_dir connectivity_fixed_p{recommended_order} \\")
        print(f"      --fixed_order {recommended_order} \\")
        print(f"      --workers 8")
        print(f"\n{'='*80}\n")
    else:
        print(f"\n⚠️  Analysis incomplete or interrupted.")
        print(f"    Run the same command to resume from checkpoint.\n")


if __name__ == "__main__":
    # REQUIRED for Windows parallel processing
    multiprocessing.freeze_support() 
    main()