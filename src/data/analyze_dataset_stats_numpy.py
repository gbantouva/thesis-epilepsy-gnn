"""
Preprocessed Dataset Statistics Analyzer
=========================================
Analyzes preprocessed EEG data (numpy files) and computes statistics.

Computes:
- Number of patients
- Number of sessions
- Number of preprocessed files (*_epochs.npy)
- Total epochs (actual count from numpy arrays)
- Epochs per file statistics

Usage:
    python analyze_preprocessed_stats.py --data_dir "F:\path\to\data_pp"
    
    # Save results to JSON
    python analyze_preprocessed_stats.py --data_dir "F:\path\to\data_pp" --output stats.json
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime


def analyze_group(group_dir, group_name):
    """
    Analyze one group (epilepsy or control) of preprocessed data.
    
    Directory structure expected:
        group_dir/
        ├── patient_id/
        │   ├── session_id/
        │   │   ├── montage/
        │   │   │   ├── *_epochs.npy
    
    Returns:
        dict with statistics
    """
    stats = {
        'group': group_name,
        'patients': set(),
        'sessions': set(),
        'files': 0,
        'total_epochs': 0,
        'epochs_per_file': [],
        'patient_epochs': defaultdict(int),
        'session_epochs': defaultdict(int),
        'failed_files': []
    }
    
    # Find all *_epochs.npy files
    epoch_files = list(group_dir.rglob('*_epochs.npy'))
    
    if len(epoch_files) == 0:
        print(f"  ⚠ No *_epochs.npy files found in {group_dir}")
        return stats
    
    print(f"\n  Analyzing {group_name}: {len(epoch_files)} epoch files...")
    
    for npy_path in tqdm(epoch_files, desc=f"  {group_name}", unit="file"):
        # Extract metadata from path
        # Expected: .../group/patient/session/montage/file_epochs.npy
        try:
            parts = npy_path.relative_to(group_dir).parts
            
            if len(parts) >= 3:
                patient_id = parts[0]
                session_id = f"{parts[0]}_{parts[1]}"  # patient_session for uniqueness
                
                stats['patients'].add(patient_id)
                stats['sessions'].add(session_id)
            else:
                # Fallback: use parent directories
                patient_id = npy_path.parent.parent.parent.name
                session_id = f"{patient_id}_{npy_path.parent.parent.name}"
                stats['patients'].add(patient_id)
                stats['sessions'].add(session_id)
        except Exception:
            patient_id = "unknown"
            session_id = "unknown"
        
        # Load numpy file and count epochs
        try:
            epochs = np.load(npy_path)
            n_epochs = epochs.shape[0]
            
            stats['files'] += 1
            stats['total_epochs'] += n_epochs
            stats['epochs_per_file'].append(n_epochs)
            stats['patient_epochs'][patient_id] += n_epochs
            stats['session_epochs'][session_id] += n_epochs
            
        except Exception as e:
            print(f"\n  ⚠ Could not load {npy_path.name}: {e}")
            stats['failed_files'].append(str(npy_path))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze preprocessed EEG dataset (numpy files) and compute statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze preprocessed data directory
    python analyze_preprocessed_stats.py --data_dir "F:\\October-Thesis\\data_pp"
    
    # Analyze balanced preprocessed data
    python analyze_preprocessed_stats.py --data_dir "F:\\October-Thesis\\data_pp_balanced"
    
    # Save results to JSON
    python analyze_preprocessed_stats.py --data_dir "F:\\October-Thesis\\data_pp" --output pp_stats.json
        """
    )
    
    parser.add_argument("--data_dir", required=True,
                        help="Root preprocessed data directory containing 00_epilepsy and 01_no_epilepsy folders")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results (optional)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Check directory exists
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return
    
    print("=" * 80)
    print("PREPROCESSED DATASET STATISTICS ANALYZER")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    # Find group directories
    epilepsy_dir = None
    control_dir = None
    
    for name in ['00_epilepsy', 'epilepsy', '00_Epilepsy']:
        if (data_dir / name).exists():
            epilepsy_dir = data_dir / name
            break
    
    for name in ['01_no_epilepsy', 'no_epilepsy', '01_No_Epilepsy', 'control']:
        if (data_dir / name).exists():
            control_dir = data_dir / name
            break
    
    # If not found, list what's available
    if epilepsy_dir is None or control_dir is None:
        print("\nAvailable subdirectories:")
        for item in data_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        
        if epilepsy_dir is None:
            epilepsy_input = input("\nEnter epilepsy folder name: ").strip()
            epilepsy_dir = data_dir / epilepsy_input
        
        if control_dir is None:
            control_input = input("Enter control folder name: ").strip()
            control_dir = data_dir / control_input
    
    print(f"\nEpilepsy directory: {epilepsy_dir}")
    print(f"Control directory:  {control_dir}")
    
    # Analyze each group
    results = {}
    
    if epilepsy_dir and epilepsy_dir.exists():
        results['epilepsy'] = analyze_group(epilepsy_dir, 'Epilepsy')
    else:
        print(f"⚠ Epilepsy directory not found: {epilepsy_dir}")
        results['epilepsy'] = None
    
    if control_dir and control_dir.exists():
        results['control'] = analyze_group(control_dir, 'Control')
    else:
        print(f"⚠ Control directory not found: {control_dir}")
        results['control'] = None
    
    # Compute summary statistics
    print("\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    def safe_stats(stats):
        if stats is None:
            return {
                'patients': 0,
                'sessions': 0,
                'files': 0,
                'total_epochs': 0,
                'avg_epochs_per_file': 0,
                'min_epochs': 0,
                'max_epochs': 0,
                'std_epochs': 0
            }
        
        epochs_list = stats['epochs_per_file']
        
        return {
            'patients': len(stats['patients']),
            'sessions': len(stats['sessions']),
            'files': stats['files'],
            'total_epochs': stats['total_epochs'],
            'avg_epochs_per_file': np.mean(epochs_list) if epochs_list else 0,
            'min_epochs': min(epochs_list) if epochs_list else 0,
            'max_epochs': max(epochs_list) if epochs_list else 0,
            'std_epochs': np.std(epochs_list) if epochs_list else 0
        }
    
    epi_stats = safe_stats(results['epilepsy'])
    ctrl_stats = safe_stats(results['control'])
    
    # Totals
    total_patients = epi_stats['patients'] + ctrl_stats['patients']
    total_sessions = epi_stats['sessions'] + ctrl_stats['sessions']
    total_files = epi_stats['files'] + ctrl_stats['files']
    total_epochs = epi_stats['total_epochs'] + ctrl_stats['total_epochs']
    avg_epochs = total_epochs / total_files if total_files > 0 else 0
    
    # Print main table
    print(f"\n{'Metric':<25} {'Epilepsy':>15} {'Control':>15} {'Total':>15}")
    print("-" * 72)
    print(f"{'Number of Patients':<25} {epi_stats['patients']:>15} {ctrl_stats['patients']:>15} {total_patients:>15}")
    print(f"{'Number of Sessions':<25} {epi_stats['sessions']:>15} {ctrl_stats['sessions']:>15} {total_sessions:>15}")
    print(f"{'Number of Files':<25} {epi_stats['files']:>15} {ctrl_stats['files']:>15} {total_files:>15}")
    print(f"{'Total Epochs':<25} {epi_stats['total_epochs']:>15,} {ctrl_stats['total_epochs']:>15,} {total_epochs:>15,}")
    print(f"{'Avg Epochs/File':<25} {epi_stats['avg_epochs_per_file']:>15.1f} {ctrl_stats['avg_epochs_per_file']:>15.1f} {avg_epochs:>15.1f}")
    print("-" * 72)
    
    # Additional statistics
    print(f"\n📊 Epoch Statistics per File:")
    
    if epi_stats['files'] > 0:
        print(f"\n   Epilepsy:")
        print(f"      Min epochs/file:  {epi_stats['min_epochs']}")
        print(f"      Max epochs/file:  {epi_stats['max_epochs']}")
        print(f"      Mean epochs/file: {epi_stats['avg_epochs_per_file']:.1f}")
        print(f"      Std epochs/file:  {epi_stats['std_epochs']:.1f}")
    
    if ctrl_stats['files'] > 0:
        print(f"\n   Control:")
        print(f"      Min epochs/file:  {ctrl_stats['min_epochs']}")
        print(f"      Max epochs/file:  {ctrl_stats['max_epochs']}")
        print(f"      Mean epochs/file: {ctrl_stats['avg_epochs_per_file']:.1f}")
        print(f"      Std epochs/file:  {ctrl_stats['std_epochs']:.1f}")
    
    # Class balance
    if ctrl_stats['total_epochs'] > 0:
        print(f"\n📊 Class Balance:")
        print(f"   Epilepsy epochs: {epi_stats['total_epochs']:,}")
        print(f"   Control epochs:  {ctrl_stats['total_epochs']:,}")
        print(f"   Ratio: {epi_stats['total_epochs']/ctrl_stats['total_epochs']:.2f}:1")
    
    # Duration estimate (assuming 4-sec epochs at 250Hz)
    print(f"\n📊 Duration Estimate (assuming 4-sec epochs):")
    epi_hours = (epi_stats['total_epochs'] * 4) / 3600
    ctrl_hours = (ctrl_stats['total_epochs'] * 4) / 3600
    total_hours = epi_hours + ctrl_hours
    print(f"   Epilepsy: {epi_hours:.2f} hours")
    print(f"   Control:  {ctrl_hours:.2f} hours")
    print(f"   Total:    {total_hours:.2f} hours")
    
    # Print for easy copy-paste
    print("\n")
    print("=" * 80)
    print("COPY-PASTE FOR DOCUMENT (Table Format)")
    print("=" * 80)
    print(f"""
| Metric                  | Epilepsy       | Control        | Total          |
|-------------------------|----------------|----------------|----------------|
| Number of Patients      | {epi_stats['patients']:<14} | {ctrl_stats['patients']:<14} | {total_patients:<14} |
| Number of Sessions      | {epi_stats['sessions']:<14} | {ctrl_stats['sessions']:<14} | {total_sessions:<14} |
| Number of Files         | {epi_stats['files']:<14} | {ctrl_stats['files']:<14} | {total_files:<14} |
| Total Epochs            | {epi_stats['total_epochs']:<14,} | {ctrl_stats['total_epochs']:<14,} | {total_epochs:<14,} |
| Avg Epochs/File         | {epi_stats['avg_epochs_per_file']:<14.1f} | {ctrl_stats['avg_epochs_per_file']:<14.1f} | {avg_epochs:<14.1f} |
""")
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': str(data_dir),
            'epilepsy': {
                'patients': epi_stats['patients'],
                'sessions': epi_stats['sessions'],
                'files': epi_stats['files'],
                'total_epochs': epi_stats['total_epochs'],
                'avg_epochs_per_file': round(epi_stats['avg_epochs_per_file'], 2),
                'min_epochs_per_file': epi_stats['min_epochs'],
                'max_epochs_per_file': epi_stats['max_epochs']
            },
            'control': {
                'patients': ctrl_stats['patients'],
                'sessions': ctrl_stats['sessions'],
                'files': ctrl_stats['files'],
                'total_epochs': ctrl_stats['total_epochs'],
                'avg_epochs_per_file': round(ctrl_stats['avg_epochs_per_file'], 2),
                'min_epochs_per_file': ctrl_stats['min_epochs'],
                'max_epochs_per_file': ctrl_stats['max_epochs']
            },
            'total': {
                'patients': total_patients,
                'sessions': total_sessions,
                'files': total_files,
                'total_epochs': total_epochs,
                'avg_epochs_per_file': round(avg_epochs, 2)
            },
            'class_balance': {
                'epilepsy_epochs': epi_stats['total_epochs'],
                'control_epochs': ctrl_stats['total_epochs'],
                'ratio': round(epi_stats['total_epochs'] / ctrl_stats['total_epochs'], 2) if ctrl_stats['total_epochs'] > 0 else None
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Results saved to: {args.output}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()