"""
Select Balanced Sessions from RAW EDF Files
============================================

This script:
1. Loads the selection plan (how many sessions per patient)
2. Finds RAW EDF files in data_raw/DATA/00_epilepsy
3. Groups EDF files by session
4. Randomly selects sessions based on the plan
5. Saves list to balanced_epilepsy_edf_files.txt

This list will be used for preprocessing ONLY the selected sessions!
"""

from pathlib import Path
import random
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to RAW epilepsy data directory
RAW_DATA_DIR = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_raw\DATA\00_epilepsy')

# Path to selection plan (from create_balanced_plan.py)
PLAN_FILE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\analysis_results\balanced_selection_stats.json')

# Output file
OUTPUT_FILE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\analysis_results\balanced_epilepsy_edf_files.txt')

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# SCRIPT
# ============================================================================

def get_session_id(edf_file):
    """
    Extract session ID from EDF file path.
    
    Example:
        Input:  .../aaaaaanr/s003_2013/01_tcp_ar/aaaaaanr_s003_t000.edf
        Output: s003_2013
    """
    # Get parent directories
    parts = edf_file.parts
    # Find session directory (e.g., s003_2013)
    for part in reversed(parts):
        if part.startswith('s') and '_' in part:
            return part
    return None


def main():
    print("="*80)
    print("SELECTING BALANCED SESSIONS FROM RAW EDF FILES")
    print("="*80)
    
    # Load selection plan
    print(f"\n1. Loading selection plan from: {PLAN_FILE}")
    if not PLAN_FILE.exists():
        print(f"   ✗ ERROR: Plan file not found!")
        print(f"   Please run create_balanced_plan.py first.")
        return
    
    with open(PLAN_FILE, 'r') as f:
        plan = json.load(f)
    
    print(f"   ✓ Loaded plan for {plan['total_patients']} patients")
    print(f"   ✓ Target: {plan['total_selected_sessions']} sessions")
    print(f"   ✓ Estimated: {plan['estimated_total_hours']:.1f} hours")
    
    # Set random seed
    random.seed(RANDOM_SEED)
    print(f"\n2. Random seed: {RANDOM_SEED} (reproducible)")
    
    # Check if data directory exists
    print(f"\n3. Scanning RAW data: {RAW_DATA_DIR}")
    if not RAW_DATA_DIR.exists():
        print(f"\n   ✗ ERROR: Directory not found!")
        print(f"   Please check the path in the script configuration.")
        return
    
    print(f"   ✓ Directory found")
    
    # Process each patient
    print(f"\n4. Selecting sessions per patient...")
    print(f"   {'Patient':<15} {'Total':>6} {'Select':>6} {'Sessions Found':>15} {'Selected':>9} {'EDF Files':>10}")
    print(f"   {'-'*80}")
    
    selected_files = []
    per_patient_selection = plan['per_patient_selection']
    
    for patient_id, info in per_patient_selection.items():
        patient_dir = RAW_DATA_DIR / patient_id
        
        # Check if patient directory exists
        if not patient_dir.exists():
            print(f"   {patient_id:<15} {'?':>6} {info['selected_sessions']:>6} {'NOT FOUND':>15} {'SKIP':>9} {0:>10}")
            continue
        
        # Find all EDF files for this patient
        edf_files = list(patient_dir.rglob('*.edf'))
        
        if len(edf_files) == 0:
            print(f"   {patient_id:<15} {info['total_sessions']:>6} {info['selected_sessions']:>6} {'NO FILES':>15} {'SKIP':>9} {0:>10}")
            continue
        
        # Group EDF files by session
        sessions_dict = {}
        for edf_file in edf_files:
            session_id = get_session_id(edf_file)
            if session_id:
                if session_id not in sessions_dict:
                    sessions_dict[session_id] = []
                sessions_dict[session_id].append(edf_file)
        
        n_sessions_found = len(sessions_dict)
        
        # Randomly select sessions
        n_to_select = min(info['selected_sessions'], n_sessions_found)
        selected_session_ids = random.sample(list(sessions_dict.keys()), k=n_to_select)
        
        # Add ALL EDF files from selected sessions
        n_files_added = 0
        for session_id in selected_session_ids:
            for file_path in sessions_dict[session_id]:
                selected_files.append(str(file_path))
                n_files_added += 1
        
        print(f"   {patient_id:<15} {info['total_sessions']:>6} {info['selected_sessions']:>6} "
              f"{n_sessions_found:>15} {n_to_select:>9} {n_files_added:>10}")
    
    # Save list
    print(f"\n5. Saving selection list...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for filepath in sorted(selected_files):
            f.write(filepath + '\n')
    
    print(f"   ✓ Saved {len(selected_files)} EDF files to: {OUTPUT_FILE}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SELECTION SUMMARY")
    print(f"{'='*80}")
    print(f"Selected EDF files:    {len(selected_files)}")
    print(f"Expected sessions:     {plan['total_selected_sessions']}")
    print(f"")
    print(f"Note: Number of EDF files > sessions because:")
    print(f"  - Each session contains multiple EDF files (different time segments)")
    print(f"  - Each session may have multiple recording montages")
    print(f"  - We select entire SESSIONS, preserving temporal continuity")
    print(f"")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"")
    print(f"Next steps:")
    print(f"  1. Run preprocessing ONLY on these selected EDF files")
    print(f"  2. This will create a balanced preprocessed dataset")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()