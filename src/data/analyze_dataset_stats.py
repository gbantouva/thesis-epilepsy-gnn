"""
Dataset Statistics Analyzer
============================
Analyzes raw EEG data folder structure and computes statistics for thesis documentation.

Computes:
- Number of patients
- Number of sessions
- Number of EDF files
- Total duration (hours)
- Average session duration
- Epoch duration (configurable)
- Total epochs (estimated)

Usage:
    python analyze_dataset_stats.py --data_dir "F:\path\to\DATA"
    
    # Or with custom epoch duration
    python analyze_dataset_stats.py --data_dir "F:\path\to\DATA" --epoch_duration 4.0
"""

import argparse
from pathlib import Path
from collections import defaultdict
import mne
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime


def get_edf_duration(edf_path):
    """
    Get duration of EDF file in seconds without loading all data.
    
    Returns:
        duration in seconds, or None if file cannot be read
    """
    try:
        # Read only header (much faster than loading data)
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose='ERROR')
        duration = raw.n_times / raw.info['sfreq']
        raw.close()
        return duration
    except Exception as e:
        print(f"  ⚠ Could not read {edf_path.name}: {e}")
        return None


def analyze_group(group_dir, group_name, epoch_duration=4.0):
    """
    Analyze one group (epilepsy or control).
    
    Directory structure expected:
        group_dir/
        ├── patient_id/
        │   ├── session_id/
        │   │   ├── montage/
        │   │   │   ├── *.edf
    
    Returns:
        dict with statistics
    """
    stats = {
        'group': group_name,
        'patients': set(),
        'sessions': set(),
        'files': 0,
        'total_duration_sec': 0.0,
        'file_durations': [],
        'session_durations': defaultdict(float),
        'patient_sessions': defaultdict(set),
        'failed_files': []
    }
    
    # Find all EDF files
    edf_files = list(group_dir.rglob('*.edf'))
    
    if len(edf_files) == 0:
        print(f"  ⚠ No EDF files found in {group_dir}")
        return stats
    
    print(f"\n  Analyzing {group_name}: {len(edf_files)} EDF files...")
    
    for edf_path in tqdm(edf_files, desc=f"  {group_name}", unit="file"):
        # Extract metadata from path
        # Expected: .../group/patient/session/montage/file.edf
        try:
            parts = edf_path.relative_to(group_dir).parts
            
            if len(parts) >= 3:
                patient_id = parts[0]
                session_id = f"{parts[0]}_{parts[1]}"  # patient_session for uniqueness
                
                stats['patients'].add(patient_id)
                stats['sessions'].add(session_id)
                stats['patient_sessions'][patient_id].add(parts[1])
            else:
                # Fallback: use parent directories
                patient_id = edf_path.parent.parent.parent.name
                session_id = f"{patient_id}_{edf_path.parent.parent.name}"
                stats['patients'].add(patient_id)
                stats['sessions'].add(session_id)
        except Exception:
            pass
        
        # Get file duration
        duration = get_edf_duration(edf_path)
        
        if duration is not None:
            stats['files'] += 1
            stats['total_duration_sec'] += duration
            stats['file_durations'].append(duration)
            
            # Track per-session duration
            if 'session_id' in dir():
                stats['session_durations'][session_id] += duration
        else:
            stats['failed_files'].append(str(edf_path))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze raw EEG dataset and compute statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze data directory
    python analyze_dataset_stats.py --data_dir "F:\\October-Thesis\\data_raw\\DATA"
    
    # With custom epoch duration
    python analyze_dataset_stats.py --data_dir "F:\\October-Thesis\\data_raw\\DATA" --epoch_duration 4.0
    
    # Save results to JSON
    python analyze_dataset_stats.py --data_dir "F:\\October-Thesis\\data_raw\\DATA" --output stats.json
        """
    )
    
    parser.add_argument("--data_dir", required=True,
                        help="Root data directory containing 00_epilepsy and 01_no_epilepsy folders")
    parser.add_argument("--epoch_duration", type=float, default=4.0,
                        help="Epoch duration in seconds (default: 4.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results (optional)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    epoch_duration = args.epoch_duration
    
    # Check directory exists
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return
    
    print("=" * 80)
    print("DATASET STATISTICS ANALYZER")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Epoch duration: {epoch_duration} seconds")
    print("=" * 80)
    
    # Find group directories
    # Try common naming conventions
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
    
    if epilepsy_dir.exists():
        results['epilepsy'] = analyze_group(epilepsy_dir, 'Epilepsy', epoch_duration)
    else:
        print(f"⚠ Epilepsy directory not found: {epilepsy_dir}")
        results['epilepsy'] = None
    
    if control_dir.exists():
        results['control'] = analyze_group(control_dir, 'Control', epoch_duration)
    else:
        print(f"⚠ Control directory not found: {control_dir}")
        results['control'] = None
    
    # Compute summary statistics
    print("\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    # Create summary table
    def safe_stats(stats):
        if stats is None:
            return {
                'patients': 0,
                'sessions': 0,
                'files': 0,
                'total_hours': 0,
                'avg_session_hours': 0,
                'total_epochs': 0
            }
        
        total_hours = stats['total_duration_sec'] / 3600
        n_sessions = len(stats['sessions'])
        avg_session_hours = total_hours / n_sessions if n_sessions > 0 else 0
        total_epochs = int(stats['total_duration_sec'] // epoch_duration)
        
        return {
            'patients': len(stats['patients']),
            'sessions': n_sessions,
            'files': stats['files'],
            'total_hours': total_hours,
            'avg_session_hours': avg_session_hours,
            'avg_session_min': avg_session_hours * 60,
            'total_epochs': total_epochs
        }
    
    epi_stats = safe_stats(results['epilepsy'])
    ctrl_stats = safe_stats(results['control'])
    
    # Totals
    total_patients = epi_stats['patients'] + ctrl_stats['patients']
    total_sessions = epi_stats['sessions'] + ctrl_stats['sessions']
    total_files = epi_stats['files'] + ctrl_stats['files']
    total_hours = epi_stats['total_hours'] + ctrl_stats['total_hours']
    total_epochs = epi_stats['total_epochs'] + ctrl_stats['total_epochs']
    avg_session_hours = total_hours / total_sessions if total_sessions > 0 else 0
    
    # Print table
    print(f"\n{'Metric':<25} {'Epilepsy':>15} {'Control':>15} {'Total':>15}")
    print("-" * 72)
    print(f"{'Number of Patients':<25} {epi_stats['patients']:>15} {ctrl_stats['patients']:>15} {total_patients:>15}")
    print(f"{'Number of Sessions':<25} {epi_stats['sessions']:>15} {ctrl_stats['sessions']:>15} {total_sessions:>15}")
    print(f"{'Number of EDF Files':<25} {epi_stats['files']:>15} {ctrl_stats['files']:>15} {total_files:>15}")
    print(f"{'Total Duration (hours)':<25} {epi_stats['total_hours']:>15.2f} {ctrl_stats['total_hours']:>15.2f} {total_hours:>15.2f}")
    print(f"{'Avg Session (minutes)':<25} {epi_stats['avg_session_min']:>15.2f} {ctrl_stats['avg_session_min']:>15.2f} {avg_session_hours*60:>15.2f}")
    print(f"{'Epoch Duration (sec)':<25} {epoch_duration:>15.1f} {epoch_duration:>15.1f} {epoch_duration:>15.1f}")
    print(f"{'Total Epochs (estimated)':<25} {epi_stats['total_epochs']:>15,} {ctrl_stats['total_epochs']:>15,} {total_epochs:>15,}")
    print("-" * 72)
    
    # Additional stats
    print(f"\n📊 Additional Statistics:")
    print(f"   Corpus size ratio (Epilepsy:Control): {epi_stats['total_hours']:.1f}:{ctrl_stats['total_hours']:.1f} hours")
    if ctrl_stats['total_hours'] > 0:
        print(f"   Imbalance ratio: {epi_stats['total_hours']/ctrl_stats['total_hours']:.2f}:1")
    
    # File duration statistics
    if results['epilepsy'] and results['epilepsy']['file_durations']:
        durations = results['epilepsy']['file_durations']
        print(f"\n   Epilepsy file durations:")
        print(f"      Min:  {min(durations)/60:.1f} min")
        print(f"      Max:  {max(durations)/60:.1f} min")
        print(f"      Mean: {np.mean(durations)/60:.1f} min")
        print(f"      Median: {np.median(durations)/60:.1f} min")
    
    if results['control'] and results['control']['file_durations']:
        durations = results['control']['file_durations']
        print(f"\n   Control file durations:")
        print(f"      Min:  {min(durations)/60:.1f} min")
        print(f"      Max:  {max(durations)/60:.1f} min")
        print(f"      Mean: {np.mean(durations)/60:.1f} min")
        print(f"      Median: {np.median(durations)/60:.1f} min")
    
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
| Number of EDF Files     | {epi_stats['files']:<14} | {ctrl_stats['files']:<14} | {total_files:<14} |
| Total Duration (hours)  | {epi_stats['total_hours']:<14.2f} | {ctrl_stats['total_hours']:<14.2f} | {total_hours:<14.2f} |
| Avg Session (minutes)   | {epi_stats['avg_session_min']:<14.2f} | {ctrl_stats['avg_session_min']:<14.2f} | {avg_session_hours*60:<14.2f} |
| Epoch Duration (sec)    | {epoch_duration:<14.1f} | {epoch_duration:<14.1f} | {epoch_duration:<14.1f} |
| Total Epochs (est.)     | {epi_stats['total_epochs']:<14,} | {ctrl_stats['total_epochs']:<14,} | {total_epochs:<14,} |
""")
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': str(data_dir),
            'epoch_duration_sec': epoch_duration,
            'epilepsy': {
                'patients': epi_stats['patients'],
                'sessions': epi_stats['sessions'],
                'files': epi_stats['files'],
                'total_duration_hours': round(epi_stats['total_hours'], 2),
                'avg_session_minutes': round(epi_stats['avg_session_min'], 2),
                'total_epochs_estimated': epi_stats['total_epochs']
            },
            'control': {
                'patients': ctrl_stats['patients'],
                'sessions': ctrl_stats['sessions'],
                'files': ctrl_stats['files'],
                'total_duration_hours': round(ctrl_stats['total_hours'], 2),
                'avg_session_minutes': round(ctrl_stats['avg_session_min'], 2),
                'total_epochs_estimated': ctrl_stats['total_epochs']
            },
            'total': {
                'patients': total_patients,
                'sessions': total_sessions,
                'files': total_files,
                'total_duration_hours': round(total_hours, 2),
                'avg_session_minutes': round(avg_session_hours * 60, 2),
                'total_epochs_estimated': total_epochs
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