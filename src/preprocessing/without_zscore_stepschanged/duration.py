import os
import mne
import json
import csv
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm

# Base directory
base_dir = r'F:\October-Thesis\thesis-epilepsy-gnn\data_raw\DATA'

def get_edf_duration(edf_path):
    """Get duration of a single EDF file in seconds"""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        duration = raw.times[-1]
        return duration
    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return 0

def format_duration(seconds):
    """Convert seconds to readable format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60
    return f"{td.days * 24 + hours:02d}:{minutes:02d}:{secs:02d}"

def analyze_dataset(base_dir, output_dir):
    """Analyze the entire TUH EEG Epilepsy Corpus"""
    
    results = {
        '00_epilepsy': {},
        '01_no_epilepsy': {},
        'summary': {}
    }
    
    # Lists to store all data for CSV files
    all_edf_files = []
    all_sessions = []
    all_patients = []
    
    groups = ['00_epilepsy', '01_no_epilepsy']
    
    for group in groups:
        group_dir = os.path.join(base_dir, group)
        
        if not os.path.exists(group_dir):
            print(f"Directory not found: {group_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {group}")
        print(f"{'='*60}")
        
        group_stats = {
            'patients': {},
            'total_patients': 0,
            'total_sessions': 0,
            'total_edf_files': 0,
            'total_duration_seconds': 0
        }
        
        # Get all patient directories
        patient_dirs = [d for d in os.listdir(group_dir) 
                       if os.path.isdir(os.path.join(group_dir, d))]
        patient_dirs.sort()
        
        group_stats['total_patients'] = len(patient_dirs)
        
        # Process each patient
        for patient_id in tqdm(patient_dirs, desc=f"Patients in {group}"):
            patient_dir = os.path.join(group_dir, patient_id)
            
            patient_stats = {
                'sessions': {},
                'total_sessions': 0,
                'total_edf_files': 0,
                'total_duration_seconds': 0
            }
            
            # Get all session directories for this patient
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
                    'total_edf_files': 0,
                    'total_duration_seconds': 0,
                    'edf_files': []
                }
                
                # Get all recording type directories (e.g., 01_tcp_ar, 02_tcp_le)
                recording_dirs = [d for d in os.listdir(session_dir) 
                                if os.path.isdir(os.path.join(session_dir, d))]
                recording_dirs.sort()
                
                # Process each recording type directory
                for recording_type in recording_dirs:
                    recording_dir = os.path.join(session_dir, recording_type)
                    
                    # Find all EDF files
                    edf_files = [f for f in os.listdir(recording_dir) 
                               if f.endswith('.edf')]
                    edf_files.sort()
                    
                    recording_stats = {
                        'edf_files': [],
                        'total_edf_files': len(edf_files),
                        'total_duration_seconds': 0
                    }
                    
                    # Process each EDF file
                    for edf_file in edf_files:
                        edf_path = os.path.join(recording_dir, edf_file)
                        duration = get_edf_duration(edf_path)
                        
                        edf_info = {
                            'filename': edf_file,
                            'duration_seconds': round(duration, 2),
                            'duration_formatted': format_duration(duration)
                        }
                        
                        recording_stats['edf_files'].append(edf_info)
                        recording_stats['total_duration_seconds'] += duration
                        session_stats['total_duration_seconds'] += duration
                        session_stats['total_edf_files'] += 1
                        
                        # Add to all_edf_files list for CSV
                        all_edf_files.append({
                            'group': group,
                            'patient_id': patient_id,
                            'session_id': session_id,
                            'recording_type': recording_type,
                            'filename': edf_file,
                            'filepath': edf_path,
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
                patient_stats['total_edf_files'] += session_stats['total_edf_files']
                
                # Add to all_sessions list for CSV
                all_sessions.append({
                    'group': group,
                    'patient_id': patient_id,
                    'session_id': session_id,
                    'num_edf_files': session_stats['total_edf_files'],
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
            group_stats['total_edf_files'] += patient_stats['total_edf_files']
            
            # Add to all_patients list for CSV
            all_patients.append({
                'group': group,
                'patient_id': patient_id,
                'num_sessions': patient_stats['total_sessions'],
                'num_edf_files': patient_stats['total_edf_files'],
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
        
        results[group] = group_stats
        
        # Print group summary
        print(f"\n{group} Summary:")
        print(f"  Total patients: {group_stats['total_patients']}")
        print(f"  Total sessions: {group_stats['total_sessions']}")
        print(f"  Total EDF files: {group_stats['total_edf_files']}")
        print(f"  Total duration: {group_stats['total_duration_formatted']} ({group_stats['total_duration_hours']} hours)")
    
    # Overall summary
    total_patients = results['00_epilepsy']['total_patients'] + results['01_no_epilepsy']['total_patients']
    total_sessions = results['00_epilepsy']['total_sessions'] + results['01_no_epilepsy']['total_sessions']
    total_edf_files = results['00_epilepsy']['total_edf_files'] + results['01_no_epilepsy']['total_edf_files']
    total_duration = results['00_epilepsy']['total_duration_seconds'] + results['01_no_epilepsy']['total_duration_seconds']
    
    results['summary'] = {
        'total_patients': total_patients,
        'total_sessions': total_sessions,
        'total_edf_files': total_edf_files,
        'total_duration_seconds': total_duration,
        'total_duration_formatted': format_duration(total_duration),
        'total_duration_hours': round(total_duration / 3600, 2),
        'epilepsy_patients': results['00_epilepsy']['total_patients'],
        'no_epilepsy_patients': results['01_no_epilepsy']['total_patients']
    }
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total patients: {total_patients}")
    print(f"  - Epilepsy: {results['00_epilepsy']['total_patients']}")
    print(f"  - No epilepsy: {results['01_no_epilepsy']['total_patients']}")
    print(f"Total sessions: {total_sessions}")
    print(f"Total EDF files: {total_edf_files}")
    print(f"Total duration: {format_duration(total_duration)} ({round(total_duration / 3600, 2)} hours)")
    
    # Save separate CSV files
    save_csv_files(all_edf_files, all_sessions, all_patients, output_dir)
    
    return results

def save_csv_files(all_edf_files, all_sessions, all_patients, output_dir):
    """Save separate CSV files for EDF files, sessions, and patients"""
    
    # Save all EDF files
    edf_csv_path = os.path.join(output_dir, 'all_edf_files.csv')
    with open(edf_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_edf_files:
            writer = csv.DictWriter(f, fieldnames=all_edf_files[0].keys())
            writer.writeheader()
            writer.writerows(all_edf_files)
    print(f"\nAll EDF files saved to: {edf_csv_path}")
    
    # Save all sessions
    sessions_csv_path = os.path.join(output_dir, 'all_sessions.csv')
    with open(sessions_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_sessions:
            writer = csv.DictWriter(f, fieldnames=all_sessions[0].keys())
            writer.writeheader()
            writer.writerows(all_sessions)
    print(f"All sessions saved to: {sessions_csv_path}")
    
    # Save all patients
    patients_csv_path = os.path.join(output_dir, 'all_patients.csv')
    with open(patients_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_patients:
            writer = csv.DictWriter(f, fieldnames=all_patients[0].keys())
            writer.writeheader()
            writer.writerows(all_patients)
    print(f"All patients saved to: {patients_csv_path}")
    
    # Save separate files per group
    for group in ['00_epilepsy', '01_no_epilepsy']:
        # EDF files per group
        group_edf_files = [edf for edf in all_edf_files if edf['group'] == group]
        group_edf_path = os.path.join(output_dir, f'{group}_edf_files.csv')
        with open(group_edf_path, 'w', newline='', encoding='utf-8') as f:
            if group_edf_files:
                writer = csv.DictWriter(f, fieldnames=group_edf_files[0].keys())
                writer.writeheader()
                writer.writerows(group_edf_files)
        print(f"{group} EDF files saved to: {group_edf_path}")
        
        # Sessions per group
        group_sessions = [sess for sess in all_sessions if sess['group'] == group]
        group_sessions_path = os.path.join(output_dir, f'{group}_sessions.csv')
        with open(group_sessions_path, 'w', newline='', encoding='utf-8') as f:
            if group_sessions:
                writer = csv.DictWriter(f, fieldnames=group_sessions[0].keys())
                writer.writeheader()
                writer.writerows(group_sessions)
        print(f"{group} sessions saved to: {group_sessions_path}")
        
        # Patients per group
        group_patients = [pat for pat in all_patients if pat['group'] == group]
        group_patients_path = os.path.join(output_dir, f'{group}_patients.csv')
        with open(group_patients_path, 'w', newline='', encoding='utf-8') as f:
            if group_patients:
                writer = csv.DictWriter(f, fieldnames=group_patients[0].keys())
                writer.writeheader()
                writer.writerows(group_patients)
        print(f"{group} patients saved to: {group_patients_path}")

def save_results(results, output_dir='./'):
    """Save results to JSON and text files"""
    
    # Save detailed JSON
    json_path = os.path.join(output_dir, 'dataset_analysis_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")
    
    # Save summary text file
    txt_path = os.path.join(output_dir, 'dataset_analysis_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("TUH EEG EPILEPSY CORPUS ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Total patients: {results['summary']['total_patients']}\n")
        f.write(f"  - Epilepsy: {results['summary']['epilepsy_patients']}\n")
        f.write(f"  - No epilepsy: {results['summary']['no_epilepsy_patients']}\n")
        f.write(f"Total sessions: {results['summary']['total_sessions']}\n")
        f.write(f"Total EDF files: {results['summary']['total_edf_files']}\n")
        f.write(f"Total duration: {results['summary']['total_duration_formatted']} ")
        f.write(f"({results['summary']['total_duration_hours']} hours)\n\n")
        
        # Group summaries
        for group in ['00_epilepsy', '01_no_epilepsy']:
            f.write(f"\n{group.upper()}\n")
            f.write("-"*60 + "\n")
            f.write(f"Patients: {results[group]['total_patients']}\n")
            f.write(f"Sessions: {results[group]['total_sessions']}\n")
            f.write(f"EDF files: {results[group]['total_edf_files']}\n")
            f.write(f"Duration: {results[group]['total_duration_formatted']} ")
            f.write(f"({results[group]['total_duration_hours']} hours)\n")
            
            # Patient details
            f.write(f"\nPatient breakdown:\n")
            for patient_id, patient_data in results[group]['patients'].items():
                f.write(f"  {patient_id}:\n")
                f.write(f"    Sessions: {patient_data['total_sessions']}\n")
                f.write(f"    EDF files: {patient_data['total_edf_files']}\n")
                f.write(f"    Duration: {patient_data['total_duration_formatted']}\n")
    
    print(f"Summary saved to: {txt_path}")
    
    # Save group statistics JSON
    group_stats_path = os.path.join(output_dir, 'group_statistics.json')
    group_stats = {
        '00_epilepsy': {
            'total_patients': results['00_epilepsy']['total_patients'],
            'total_sessions': results['00_epilepsy']['total_sessions'],
            'total_edf_files': results['00_epilepsy']['total_edf_files'],
            'total_duration_seconds': results['00_epilepsy']['total_duration_seconds'],
            'total_duration_hours': results['00_epilepsy']['total_duration_hours'],
            'total_duration_formatted': results['00_epilepsy']['total_duration_formatted']
        },
        '01_no_epilepsy': {
            'total_patients': results['01_no_epilepsy']['total_patients'],
            'total_sessions': results['01_no_epilepsy']['total_sessions'],
            'total_edf_files': results['01_no_epilepsy']['total_edf_files'],
            'total_duration_seconds': results['01_no_epilepsy']['total_duration_seconds'],
            'total_duration_hours': results['01_no_epilepsy']['total_duration_hours'],
            'total_duration_formatted': results['01_no_epilepsy']['total_duration_formatted']
        },
        'overall': results['summary']
    }
    with open(group_stats_path, 'w') as f:
        json.dump(group_stats, f, indent=2)
    print(f"Group statistics saved to: {group_stats_path}")

if __name__ == "__main__":
    # Run the analysis
    output_dir = r'F:\October-Thesis\thesis-epilepsy-gnn\analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    results = analyze_dataset(base_dir, output_dir)
    
    # Save results
    save_results(results, output_dir)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nGenerated files:")
    print("1. all_edf_files.csv - All EDF files with durations")
    print("2. all_sessions.csv - All sessions with durations")
    print("3. all_patients.csv - All patients with durations")
    print("4. 00_epilepsy_edf_files.csv - Epilepsy EDF files")
    print("5. 00_epilepsy_sessions.csv - Epilepsy sessions")
    print("6. 00_epilepsy_patients.csv - Epilepsy patients")
    print("7. 01_no_epilepsy_edf_files.csv - No epilepsy EDF files")
    print("8. 01_no_epilepsy_sessions.csv - No epilepsy sessions")
    print("9. 01_no_epilepsy_patients.csv - No epilepsy patients")
    print("10. dataset_analysis_detailed.json - Complete hierarchical data")
    print("11. dataset_analysis_summary.txt - Human-readable summary")
    print("12. group_statistics.json - Summary statistics per group")