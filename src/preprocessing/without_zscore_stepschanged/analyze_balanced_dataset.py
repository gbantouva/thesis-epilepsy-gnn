"""
Analyze Balanced Dataset (Preprocessed Epochs)
==============================================

This script analyzes the balanced dataset by reading *_epochs.npy files
and calculating duration based on number of epochs.

Analyzes both groups:
- 00_epilepsy (balanced - selected sessions)
- 01_no_epilepsy (complete - all control sessions)

Assumptions:
- Each epoch is 4 seconds long
- Epoch files have shape: (n_epochs, n_channels, n_samples)
- Structure: balanced_data/GROUP/patient/session/recording_type/*.npy
"""

import os
import numpy as np
import json
import csv
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory of balanced data (contains 00_epilepsy and 01_control)
BASE_DIR = r'F:\October-Thesis\thesis-epilepsy-gnn\data_pp\version3\balanced_data'

# Output directory for results
OUTPUT_DIR = r'F:\October-Thesis\thesis-epilepsy-gnn\balanced_analysis_results'

# Epoch duration in seconds
EPOCH_DURATION = 4  # seconds per epoch

# ============================================================================
# FUNCTIONS
# ============================================================================

def get_epochs_duration(epochs_path):
    """
    Get duration from epochs.npy file.
    
    Returns:
        tuple: (n_epochs, duration_seconds)
    """
    try:
        epochs = np.load(epochs_path, mmap_mode='r')  # Memory-mapped for speed
        n_epochs = epochs.shape[0]
        duration = n_epochs * EPOCH_DURATION
        return n_epochs, duration
    except Exception as e:
        print(f"Error reading {epochs_path}: {e}")
        return 0, 0


def format_duration(seconds):
    """Convert seconds to readable format HH:MM:SS"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{td.days * 24 + hours:02d}:{minutes:02d}:{secs:02d}"


def analyze_balanced_dataset(base_dir, output_dir):
    """Analyze the balanced preprocessed dataset"""
    
    results = {
        '00_epilepsy': {},
        '01_no_epilepsy': {},
        'summary': {}
    }
    
    # Lists for CSV output
    all_epoch_files = []
    all_sessions = []
    all_patients = []
    
    # Process both groups
    groups = ['00_epilepsy', '01_no_epilepsy']
    
    for group in groups:
        group_dir = os.path.join(base_dir, group)
        
        if not os.path.exists(group_dir):
            print(f"⚠ Directory not found (skipping): {group_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"ANALYZING: {group}")
        print(f"{'='*80}")
        
        group_stats = {
            'patients': {},
            'total_patients': 0,
            'total_sessions': 0,
            'total_epoch_files': 0,
            'total_epochs': 0,
            'total_duration_seconds': 0
        }
        
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(group_dir) 
                       if os.path.isdir(os.path.join(group_dir, d))]
        patient_dirs.sort()
        
        group_stats['total_patients'] = len(patient_dirs)
        
        print(f"\nFound {len(patient_dirs)} patients")
        print(f"Epoch duration: {EPOCH_DURATION} seconds\n")
        
        # Process each patient
        for patient_id in tqdm(patient_dirs, desc=f"  {group}"):
            patient_dir = os.path.join(group_dir, patient_id)
            patient_dir = os.path.join(group_dir, patient_id)
            
            patient_stats = {
                'sessions': {},
                'total_sessions': 0,
                'total_epoch_files': 0,
                'total_epochs': 0,
                'total_duration_seconds': 0
            }
            
            # Get all session directories
            session_dirs = [d for d in os.listdir(patient_dir) 
                          if os.path.isdir(os.path.join(patient_dir, d))]
            session_dirs.sort()
            
            patient_stats['total_sessions'] = len(session_dirs)
            group_stats['total_sessions'] += len(session_dirs)
            
            # Process each session
            for session_id in session_dirs:
                session_dir = os.path.join(patient_dir, session_id)
                
                session_stats = {
                    'recording_types': {},
                    'total_epoch_files': 0,
                    'total_epochs': 0,
                    'total_duration_seconds': 0,
                    'epoch_files': []
                }
                
                # Get all recording type directories
                recording_dirs = [d for d in os.listdir(session_dir) 
                                if os.path.isdir(os.path.join(session_dir, d))]
                recording_dirs.sort()
                
                # Process each recording type
                for recording_type in recording_dirs:
                    recording_dir = os.path.join(session_dir, recording_type)
                    
                    # Find all epochs.npy files
                    epoch_files = [f for f in os.listdir(recording_dir) 
                                 if f.endswith('_epochs.npy')]
                    epoch_files.sort()
                    
                    recording_stats = {
                        'epoch_files': [],
                        'total_epoch_files': len(epoch_files),
                        'total_epochs': 0,
                        'total_duration_seconds': 0
                    }
                    
                    # Process each epochs.npy file
                    for epoch_file in epoch_files:
                        epoch_path = os.path.join(recording_dir, epoch_file)
                        n_epochs, duration = get_epochs_duration(epoch_path)
                        
                        epoch_info = {
                            'filename': epoch_file,
                            'n_epochs': n_epochs,
                            'duration_seconds': round(duration, 2),
                            'duration_formatted': format_duration(duration)
                        }
                        
                        recording_stats['epoch_files'].append(epoch_info)
                        recording_stats['total_epochs'] += n_epochs
                        recording_stats['total_duration_seconds'] += duration
                        session_stats['total_duration_seconds'] += duration
                        session_stats['total_epochs'] += n_epochs
                        session_stats['total_epoch_files'] += 1
                        
                        # Add to all_epoch_files list for CSV
                        all_epoch_files.append({
                            'group': group,
                            'patient_id': patient_id,
                            'session_id': session_id,
                            'recording_type': recording_type,
                            'filename': epoch_file,
                            'filepath': epoch_path,
                            'n_epochs': n_epochs,
                            'duration_seconds': round(duration, 2),
                            'duration_formatted': format_duration(duration),
                            'duration_minutes': round(duration / 60, 2),
                            'duration_hours': round(duration / 3600, 2)
                        })
                    
                    recording_stats['total_duration_formatted'] = format_duration(
                        recording_stats['total_duration_seconds']
                    )
                    session_stats['recording_types'][recording_type] = recording_stats
                
                # Update session stats
                session_stats['total_duration_formatted'] = format_duration(
                    session_stats['total_duration_seconds']
                )
                patient_stats['sessions'][session_id] = session_stats
                patient_stats['total_duration_seconds'] += session_stats['total_duration_seconds']
                patient_stats['total_epochs'] += session_stats['total_epochs']
                patient_stats['total_epoch_files'] += session_stats['total_epoch_files']
                
                # Add to all_sessions list for CSV
                all_sessions.append({
                    'group': group,
                    'patient_id': patient_id,
                    'session_id': session_id,
                    'num_epoch_files': session_stats['total_epoch_files'],
                    'total_epochs': session_stats['total_epochs'],
                    'duration_seconds': round(session_stats['total_duration_seconds'], 2),
                    'duration_formatted': session_stats['total_duration_formatted'],
                    'duration_minutes': round(session_stats['total_duration_seconds'] / 60, 2),
                    'duration_hours': round(session_stats['total_duration_seconds'] / 3600, 2)
                })
            
            # Update patient stats
            patient_stats['total_duration_formatted'] = format_duration(
                patient_stats['total_duration_seconds']
            )
            group_stats['patients'][patient_id] = patient_stats
            group_stats['total_duration_seconds'] += patient_stats['total_duration_seconds']
            group_stats['total_epochs'] += patient_stats['total_epochs']
            group_stats['total_epoch_files'] += patient_stats['total_epoch_files']
            
            # Add to all_patients list for CSV
            all_patients.append({
                'group': group,
                'patient_id': patient_id,
                'num_sessions': patient_stats['total_sessions'],
                'num_epoch_files': patient_stats['total_epoch_files'],
                'total_epochs': patient_stats['total_epochs'],
                'duration_seconds': round(patient_stats['total_duration_seconds'], 2),
                'duration_formatted': patient_stats['total_duration_formatted'],
                'duration_minutes': round(patient_stats['total_duration_seconds'] / 60, 2),
                'duration_hours': round(patient_stats['total_duration_seconds'] / 3600, 2)
            })
        
        # Format group totals
        group_stats['total_duration_formatted'] = format_duration(
            group_stats['total_duration_seconds']
        )
        group_stats['total_duration_hours'] = round(
            group_stats['total_duration_seconds'] / 3600, 2
        )
        group_stats['total_duration_minutes'] = round(
            group_stats['total_duration_seconds'] / 60, 2
        )
        
        results[group] = group_stats
        
        # Print group summary
        print(f"\n{group} Summary:")
        print(f"  Patients:     {group_stats['total_patients']}")
        print(f"  Sessions:     {group_stats['total_sessions']}")
        print(f"  Epoch files:  {group_stats['total_epoch_files']}")
        print(f"  Total epochs: {group_stats['total_epochs']:,}")
        print(f"  Duration:     {group_stats['total_duration_formatted']} ({group_stats['total_duration_hours']:.2f} hours)")
    
    # Overall summary across both groups
    total_patients = sum(results.get(g, {}).get('total_patients', 0) for g in groups)
    total_sessions = sum(results.get(g, {}).get('total_sessions', 0) for g in groups)
    total_epoch_files = sum(results.get(g, {}).get('total_epoch_files', 0) for g in groups)
    total_epochs = sum(results.get(g, {}).get('total_epochs', 0) for g in groups)
    total_duration = sum(results.get(g, {}).get('total_duration_seconds', 0) for g in groups)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BALANCED DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal patients:     {total_patients}")
    print(f"Total sessions:     {total_sessions}")
    print(f"Total epoch files:  {total_epoch_files}")
    print(f"Total epochs:       {total_epochs:,}")
    print(f"Total duration:     {format_duration(total_duration)} ({total_duration / 3600:.2f} hours)")
    
    # Calculate balance ratio
    epilepsy_hours = results.get('00_epilepsy', {}).get('total_duration_hours', 0)
    control_hours = results.get('01_no_epilepsy', {}).get('total_duration_hours', 0)
    
    if control_hours > 0:
        balance_ratio = epilepsy_hours / control_hours
        epilepsy_pct = (epilepsy_hours / (epilepsy_hours + control_hours)) * 100
        control_pct = (control_hours / (epilepsy_hours + control_hours)) * 100
        
        print(f"\nBalance Analysis:")
        print(f"  Epilepsy:  {epilepsy_hours:>7.2f} hours ({epilepsy_pct:>5.1f}%)")
        print(f"  Control:   {control_hours:>7.2f} hours ({control_pct:>5.1f}%)")
        print(f"  Ratio:     {balance_ratio:.2f}:1 (epilepsy:control)")
    
    # Summary stats
    results['summary'] = {
        'total_patients': total_patients,
        'total_sessions': total_sessions,
        'total_epoch_files': total_epoch_files,
        'total_epochs': total_epochs,
        'total_duration_seconds': total_duration,
        'total_duration_formatted': format_duration(total_duration),
        'total_duration_hours': round(total_duration / 3600, 2),
        'total_duration_minutes': round(total_duration / 60, 2),
        'epilepsy_hours': epilepsy_hours,
        'control_hours': control_hours,
        'balance_ratio': round(balance_ratio, 2) if control_hours > 0 else 0,
        'epilepsy_percentage': round(epilepsy_pct, 2) if control_hours > 0 else 0,
        'control_percentage': round(control_pct, 2) if control_hours > 0 else 0,
        'avg_epochs_per_file': round(total_epochs / total_epoch_files, 2) if total_epoch_files > 0 else 0,
        'avg_duration_per_session_minutes': round((total_duration / 60) / total_sessions, 2) if total_sessions > 0 else 0,
        'avg_sessions_per_patient': round(total_sessions / total_patients, 2) if total_patients > 0 else 0
    }
    
    # Save CSV files
    save_csv_files(all_epoch_files, all_sessions, all_patients, output_dir)
    
    return results


def save_csv_files(all_epoch_files, all_sessions, all_patients, output_dir):
    """Save CSV files with analysis results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all epoch files
    epoch_csv_path = os.path.join(output_dir, 'balanced_epoch_files.csv')
    with open(epoch_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_epoch_files:
            writer = csv.DictWriter(f, fieldnames=all_epoch_files[0].keys())
            writer.writeheader()
            writer.writerows(all_epoch_files)
    print(f"\n✓ Epoch files saved to: {epoch_csv_path}")
    
    # Save all sessions
    sessions_csv_path = os.path.join(output_dir, 'balanced_sessions.csv')
    with open(sessions_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_sessions:
            writer = csv.DictWriter(f, fieldnames=all_sessions[0].keys())
            writer.writeheader()
            writer.writerows(all_sessions)
    print(f"✓ Sessions saved to: {sessions_csv_path}")
    
    # Save all patients
    patients_csv_path = os.path.join(output_dir, 'balanced_patients.csv')
    with open(patients_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_patients:
            writer = csv.DictWriter(f, fieldnames=all_patients[0].keys())
            writer.writeheader()
            writer.writerows(all_patients)
    print(f"✓ Patients saved to: {patients_csv_path}")


def save_results(results, output_dir):
    """Save detailed results to JSON and text files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, 'balanced_analysis_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {json_path}")
    
    # Save summary text
    txt_path = os.path.join(output_dir, 'balanced_analysis_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("BALANCED DATASET ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total patients:     {results['summary']['total_patients']}\n")
        f.write(f"Total sessions:     {results['summary']['total_sessions']}\n")
        f.write(f"Total epoch files:  {results['summary']['total_epoch_files']}\n")
        f.write(f"Total epochs:       {results['summary']['total_epochs']:,}\n")
        f.write(f"Total duration:     {results['summary']['total_duration_formatted']} ")
        f.write(f"({results['summary']['total_duration_hours']:.2f} hours)\n\n")
        
        # Balance information
        if 'balance_ratio' in results['summary'] and results['summary']['balance_ratio'] > 0:
            f.write("BALANCE ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Epilepsy:  {results['summary']['epilepsy_hours']:>7.2f} hours ({results['summary']['epilepsy_percentage']:>5.1f}%)\n")
            f.write(f"Control:   {results['summary']['control_hours']:>7.2f} hours ({results['summary']['control_percentage']:>5.1f}%)\n")
            f.write(f"Ratio:     {results['summary']['balance_ratio']:.2f}:1 (epilepsy:control)\n\n")
        
        f.write(f"Average epochs per file:        {results['summary']['avg_epochs_per_file']:.2f}\n")
        f.write(f"Average duration per session:   {results['summary']['avg_duration_per_session_minutes']:.2f} minutes\n")
        f.write(f"Average sessions per patient:   {results['summary']['avg_sessions_per_patient']:.2f}\n\n")
        
        # Group breakdowns
        for group in ['00_epilepsy', '01_no_epilepsy']:
            if group not in results or not results[group]:
                continue
                
            f.write(f"\n{group.upper()} - SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Patients:     {results[group]['total_patients']}\n")
            f.write(f"Sessions:     {results[group]['total_sessions']}\n")
            f.write(f"Epoch files:  {results[group]['total_epoch_files']}\n")
            f.write(f"Total epochs: {results[group]['total_epochs']:,}\n")
            f.write(f"Duration:     {results[group]['total_duration_formatted']} ")
            f.write(f"({results[group]['total_duration_hours']:.2f} hours)\n\n")
            
            # Patient breakdown (first 10 patients for brevity)
            f.write(f"{group.upper()} - PATIENT BREAKDOWN (first 10)\n")
            f.write("-"*80 + "\n")
            
            for idx, (patient_id, patient_data) in enumerate(list(results[group]['patients'].items())[:10]):
                f.write(f"\n{patient_id}:\n")
                f.write(f"  Sessions:     {patient_data['total_sessions']}\n")
                f.write(f"  Epoch files:  {patient_data['total_epoch_files']}\n")
                f.write(f"  Total epochs: {patient_data['total_epochs']}\n")
                f.write(f"  Duration:     {patient_data['total_duration_formatted']} ")
                f.write(f"({patient_data['total_duration_seconds'] / 3600:.2f} hours)\n")
            
            if len(results[group]['patients']) > 10:
                f.write(f"\n... and {len(results[group]['patients']) - 10} more patients\n")
    
    print(f"✓ Summary saved to: {txt_path}")
    
    # Save statistics JSON
    stats_path = os.path.join(output_dir, 'balanced_statistics.json')
    stats = {
        'dataset': 'balanced_dataset',
        'total_patients': results['summary']['total_patients'],
        'total_sessions': results['summary']['total_sessions'],
        'total_epoch_files': results['summary']['total_epoch_files'],
        'total_epochs': results['summary']['total_epochs'],
        'total_duration_seconds': results['summary']['total_duration_seconds'],
        'total_duration_hours': results['summary']['total_duration_hours'],
        'total_duration_minutes': results['summary']['total_duration_minutes'],
        'total_duration_formatted': results['summary']['total_duration_formatted'],
        'epoch_duration_seconds': EPOCH_DURATION,
        'epilepsy': {
            'hours': results['summary'].get('epilepsy_hours', 0),
            'percentage': results['summary'].get('epilepsy_percentage', 0)
        },
        'control': {
            'hours': results['summary'].get('control_hours', 0),
            'percentage': results['summary'].get('control_percentage', 0)
        },
        'balance_ratio': results['summary'].get('balance_ratio', 0),
        'averages': {
            'epochs_per_file': results['summary']['avg_epochs_per_file'],
            'duration_per_session_minutes': results['summary']['avg_duration_per_session_minutes'],
            'sessions_per_patient': results['summary']['avg_sessions_per_patient']
        }
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("BALANCED DATASET DURATION ANALYSIS")
    print("="*80)
    
    # Check if base directory exists
    if not os.path.exists(BASE_DIR):
        print(f"\n❌ ERROR: Base directory not found: {BASE_DIR}")
        print(f"\nPlease update BASE_DIR in the script to point to your balanced_data directory.")
        exit(1)
    
    # Run analysis
    results = analyze_balanced_dataset(BASE_DIR, OUTPUT_DIR)
    
    if results:
        # Save results
        save_results(results, OUTPUT_DIR)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"\nGenerated files in: {OUTPUT_DIR}")
        print("\n1. balanced_epoch_files.csv    - All epoch files with durations")
        print("2. balanced_sessions.csv       - All sessions with durations")
        print("3. balanced_patients.csv       - All patients with durations")
        print("4. balanced_analysis_detailed.json - Complete hierarchical data")
        print("5. balanced_analysis_summary.txt   - Human-readable summary")
        print("6. balanced_statistics.json    - Summary statistics")
        print(f"\n{'='*80}")