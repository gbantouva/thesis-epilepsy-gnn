"""
Create Balanced Dataset Selection Plan
=======================================

This script creates a JSON file with the selection plan:
- How many sessions to select per patient
- Target ratio and total duration

Input:  00_epilepsy_sessions.csv (from duration.py)
Output: balanced_selection_stats.json
"""

import pandas as pd
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input file from duration.py
INPUT_CSV = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\analysis_results\00_epilepsy_sessions.csv')

# Output file
OUTPUT_JSON = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\analysis_results\balanced_selection_stats.json')

# Target: Match control group duration (~101.5 hours)
# We'll select sessions to get ~100 hours ( ratio with control)
TARGET_DURATION_HOURS = 100.0

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# SCRIPT
# ============================================================================

def main():
    print("="*80)
    print("CREATING BALANCED SELECTION PLAN")
    print("="*80)
    
    # Load session data
    print(f"\n1. Loading session data from: {INPUT_CSV}")
    
    if not INPUT_CSV.exists():
        print(f"   ✗ ERROR: File not found!")
        print(f"   Please run duration.py first to generate this file.")
        return
    
    df = pd.read_csv(INPUT_CSV)
    print(f"   ✓ Loaded {len(df)} sessions")
    
    # Group by patient
    print(f"\n2. Analyzing per-patient statistics...")
    patient_stats = df.groupby('patient_id').agg({
        'session_id': 'count',
        'duration_hours': 'sum'
    }).rename(columns={
        'session_id': 'total_sessions',
        'duration_hours': 'total_duration_hours'
    })
    
    print(f"   ✓ Found {len(patient_stats)} patients")
    print(f"   ✓ Total sessions: {patient_stats['total_sessions'].sum()}")
    print(f"   ✓ Total duration: {patient_stats['total_duration_hours'].sum():.2f} hours")
    
    # Calculate target ratio
    total_duration = patient_stats['total_duration_hours'].sum()
    target_ratio = TARGET_DURATION_HOURS / total_duration
    
    print(f"\n3. Calculating selection plan...")
    print(f"   Current total: {total_duration:.2f} hours")
    print(f"   Target total:  {TARGET_DURATION_HOURS:.2f} hours")
    print(f"   Selection ratio: {target_ratio:.3f} ({target_ratio*100:.1f}%)")
    
    # Calculate sessions to select per patient
    selection_plan = {}
    total_selected_sessions = 0
    estimated_total_hours = 0.0
    
    for patient_id, stats in patient_stats.iterrows():
        total_sessions = int(stats['total_sessions'])
        total_duration_hours = float(stats['total_duration_hours'])
        
        # How many sessions to select (at least 1 if patient has sessions)
        selected_sessions = max(1, int(round(total_sessions * target_ratio)))
        
        # Ensure we don't select more than available
        selected_sessions = min(selected_sessions, total_sessions)
        
        # Estimate duration (assuming uniform distribution)
        estimated_duration = (total_duration_hours / total_sessions) * selected_sessions
        
        selection_plan[patient_id] = {
            'total_sessions': total_sessions,
            'selected_sessions': selected_sessions,
            'percentage': round((selected_sessions / total_sessions) * 100, 2),
            'total_duration_hours': round(total_duration_hours, 2),
            'estimated_duration_hours': round(estimated_duration, 2)
        }
        
        total_selected_sessions += selected_sessions
        estimated_total_hours += estimated_duration
    
    print(f"   ✓ Plan created!")
    print(f"   Selected sessions: {total_selected_sessions}")
    print(f"   Estimated duration: {estimated_total_hours:.2f} hours")
    
    # Create final JSON structure
    output_data = {
        'random_seed': RANDOM_SEED,
        'target_duration_hours': TARGET_DURATION_HOURS,
        'target_ratio': round(target_ratio, 6),
        'total_patients': len(patient_stats),
        'total_selected_sessions': total_selected_sessions,
        'estimated_total_hours': round(estimated_total_hours, 2),
        'original_total_hours': round(total_duration, 2),
        'per_patient_selection': selection_plan
    }
    
    # Save to JSON
    print(f"\n4. Saving plan to: {OUTPUT_JSON}")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Saved!")
    
    # Summary
    print(f"\n{'='*80}")
    print("PLAN SUMMARY")
    print(f"{'='*80}")
    print(f"Total patients:        {len(patient_stats)}")
    print(f"Original sessions:     {patient_stats['total_sessions'].sum()}")
    print(f"Selected sessions:     {total_selected_sessions}")
    print(f"Reduction:             {100 * (1 - total_selected_sessions / patient_stats['total_sessions'].sum()):.1f}%")
    print(f"")
    print(f"Original duration:     {total_duration:.2f} hours")
    print(f"Estimated duration:    {estimated_total_hours:.2f} hours")
    print(f"Target duration:       {TARGET_DURATION_HOURS:.2f} hours")
    print(f"")
    print(f"Control duration:      101.5 hours")
    print(f"Epilepsy (balanced):   {estimated_total_hours:.2f} hours")
    print(f"Balance ratio:         {estimated_total_hours/101.5:.2f}:1")
    print(f"")
    print(f"Output file: {OUTPUT_JSON}")
    print(f"{'='*80}")
    
    # Show a few examples
    print(f"\nExample selections:")
    for i, (patient_id, plan) in enumerate(list(selection_plan.items())[:5]):
        print(f"  {patient_id}: {plan['selected_sessions']}/{plan['total_sessions']} sessions "
              f"({plan['percentage']:.1f}%) → ~{plan['estimated_duration_hours']:.1f}h")
    print(f"  ... ({len(selection_plan) - 5} more patients)")
    
    print(f"\nNext step: Run create_balanced_sessions.py to select actual sessions!")


if __name__ == "__main__":
    main()
