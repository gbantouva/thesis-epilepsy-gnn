"""
Create Balanced Epilepsy Session List
======================================

This script:
1. Loads the selection plan (how many sessions per patient)
2. Scans your data_pp directory
3. Randomly selects sessions based on the plan
4. Saves list to balanced_epilepsy_sessions.txt

IMPORTANT: Change DATA_DIR to your actual path!
"""

from pathlib import Path
import random
import json

# ============================================================================
# CONFIGURATION - CHANGE THESE PATHS!
# ============================================================================

# Path to your epilepsy data directory
DATA_DIR = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_pp\version3\00_epilepsy')

# Path to selection plan (already created)
PLAN_FILE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\src\preprocessing\without_zscore_stepschanged\balanced_selection_stats.json')

# Output file
OUTPUT_FILE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\balanced_epilepsy_sessions.txt')

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# SCRIPT
# ============================================================================

def main():
    print("="*80)
    print("CREATING BALANCED EPILEPSY SESSION LIST")
    print("="*80)
    
    # Load selection plan
    print(f"\n1. Loading selection plan from: {PLAN_FILE}")
    with open(PLAN_FILE, 'r') as f:
        plan = json.load(f)
    
    print(f"   ✓ Loaded plan for {plan['total_patients']} patients")
    print(f"   ✓ Target: {plan['total_selected_sessions']} sessions")
    print(f"   ✓ Estimated: {plan['estimated_total_hours']:.1f} hours")
    
    # Set random seed
    random.seed(RANDOM_SEED)
    print(f"\n2. Random seed: {RANDOM_SEED} (reproducible)")
    
    # Check if data directory exists
    print(f"\n3. Scanning data directory: {DATA_DIR}")
    if not DATA_DIR.exists():
        print(f"\n   ❌ ERROR: Directory not found!")
        print(f"   Please change DATA_DIR in the script to your actual path.")
        print(f"   Example: DATA_DIR = Path('/home/user/thesis/data_pp/version3/00_epilepsy')")
        return
    
    print(f"   ✓ Directory found")
    
    # Process each patient
    print(f"\n4. Selecting sessions per patient...")
    print(f"   {'Patient':<15} {'Total':>6} {'Select':>6} {'Found':>6} {'Selected':>6}")
    print(f"   {'-'*60}")
    
    selected_files = []
    per_patient_selection = plan['per_patient_selection']
    
    for patient_id, info in per_patient_selection.items():
        patient_dir = DATA_DIR / patient_id
        
        # Check if patient directory exists
        if not patient_dir.exists():
            print(f"   {patient_id:<15} {'?':>6} {info['selected_sessions']:>6} {'N/A':>6} {'SKIP':>6} (not found)")
            continue
        
        # Find all session epoch files
        # Pattern: *_epochs.npy
        session_files = list(patient_dir.rglob('*_epochs.npy'))
        
        if len(session_files) == 0:
            print(f"   {patient_id:<15} {info['total_sessions']:>6} {info['selected_sessions']:>6} {len(session_files):>6} {'SKIP':>6} (no files)")
            continue
        
        # Randomly select sessions
        n_to_select = min(info['selected_sessions'], len(session_files))
        selected = random.sample(session_files, k=n_to_select)
        
        # Add to list
        for file_path in selected:
            selected_files.append(str(file_path))
        
        print(f"   {patient_id:<15} {info['total_sessions']:>6} {info['selected_sessions']:>6} {len(session_files):>6} {n_to_select:>6}")
    
    # Save list
    print(f"\n5. Saving selection list...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for filepath in sorted(selected_files):
            f.write(filepath + '\n')
    
    print(f"   ✓ Saved {len(selected_files)} sessions to: {OUTPUT_FILE}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUCCESS!")
    print(f"{'='*80}")
    print(f"\nSelected sessions: {len(selected_files)}")
    print(f"Expected sessions: {plan['total_selected_sessions']}")
    print(f"Match: {'✓ YES' if len(selected_files) == plan['total_selected_sessions'] else '⚠ CHECK DATA'}")
    
    print(f"\nOutput file: {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Verify the file has {len(selected_files)} lines")
    print(f"  2. Use this list in your graph construction code")
    print(f"  3. Process only sessions in this list for epilepsy group")
    print(f"  4. Process all sessions for control group")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()