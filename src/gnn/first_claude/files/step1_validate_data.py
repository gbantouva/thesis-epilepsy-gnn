"""
STEP 1: DATA VALIDATION (ADAPTED FOR PRE-COMPUTED CONNECTIVITY)
================================================================
Validates your pre-computed connectivity files and preprocessed epochs.

Your data structure:
  connectivity/january_fixed_15/
  ├── 00_epilepsy/
  │   └── patient_id/session_id/.../xxx_graphs.npz
  └── 01_control/  (or similar)
      └── patient_id/session_id/.../xxx_graphs.npz

Each .npz contains:
  - pdc_integrated, dtf_integrated (and per-band)
  - labels
  - indices
  - Shape: (n_valid_epochs, 22, 22)
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

CONNECTIVITY_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_fixed_15")
EPOCHS_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced")  # Optional

# Output
OUTPUT_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\gnn\control_vs_epilepsy")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Dataset parameters
FS = 250  # Sampling frequency
N_CHANNELS = 22  # Your data has 22 channels

# Class identification (adjust based on your directory names)
EPILEPSY_DIR_PATTERN = "00_epilepsy"  # Directory containing epilepsy patients
CONTROL_DIR_PATTERN = "01_no_epilepsy"    # Directory containing control subjects
# Alternative patterns if different:
# EPILEPSY_DIR_PATTERN = "epilepsy"
# CONTROL_DIR_PATTERN = "control"

# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset():
    """Validate pre-computed connectivity files."""
    
    print("="*80)
    print("DATA VALIDATION (PRE-COMPUTED CONNECTIVITY)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Connectivity dir: {CONNECTIVITY_DIR}")
    print(f"  Sampling rate:    {FS} Hz")
    print(f"  Channels:         {N_CHANNELS}")
    print(f"  Epilepsy pattern: '{EPILEPSY_DIR_PATTERN}'")
    print(f"  Control pattern:  '{CONTROL_DIR_PATTERN}'")
    
    # Find all .npz files
    print(f"\n[1] Scanning for connectivity files...")
    npz_files = list(CONNECTIVITY_DIR.rglob("*_graphs.npz"))
    print(f"  ✓ Found {len(npz_files)} connectivity files")
    
    if len(npz_files) == 0:
        print(f"  ❌ No files found!")
        print(f"  → Check CONNECTIVITY_DIR path")
        return False
    
    # Classify files by label
    epilepsy_files = []
    control_files = []
    unknown_files = []
    
    for f in npz_files:
        path_str = str(f)
        if EPILEPSY_DIR_PATTERN in path_str:
            epilepsy_files.append(f)
        elif CONTROL_DIR_PATTERN in path_str:
            control_files.append(f)
        else:
            unknown_files.append(f)
    
    print(f"\n[2] File classification:")
    print(f"  Epilepsy files: {len(epilepsy_files)}")
    print(f"  Control files:  {len(control_files)}")
    if unknown_files:
        print(f"  ⚠️  Unknown:      {len(unknown_files)}")
        print(f"      → Cannot determine class from path")
        print(f"      → First unknown: {unknown_files[0].relative_to(CONNECTIVITY_DIR)}")
    
    if len(epilepsy_files) == 0 or len(control_files) == 0:
        print(f"\n  ❌ Missing one class!")
        print(f"  → Check EPILEPSY_DIR_PATTERN and CONTROL_DIR_PATTERN")
        return False
    
    # Validate file structure
    print(f"\n[3] Validating file structure...")
    
    # Sample from each class
    sample_epilepsy = np.load(epilepsy_files[0])
    sample_control = np.load(control_files[0])
    
    # Check required fields
    required_fields = ['pdc_integrated', 'labels']
    
    for sample, name in [(sample_epilepsy, 'epilepsy'), (sample_control, 'control')]:
        print(f"\n  {name.upper()} sample:")
        missing = [f for f in required_fields if f not in sample.files]
        
        if missing:
            print(f"    ❌ Missing fields: {missing}")
            return False
        
        pdc = sample['pdc_integrated']
        labels = sample['labels']
        
        print(f"    ✓ PDC shape: {pdc.shape}")
        print(f"    ✓ Labels shape: {labels.shape}")
        print(f"    ✓ Epochs: {len(labels)}")
        
        # Check dimensions
        if pdc.shape[1] != N_CHANNELS or pdc.shape[2] != N_CHANNELS:
            print(f"    ❌ Expected {N_CHANNELS} channels, got {pdc.shape[1:]}!")
            return False
        
        # Check diagonal is zero
        has_nonzero_diag = np.any([np.diag(pdc[i]) != 0 for i in range(min(5, len(pdc)))])
        if has_nonzero_diag:
            print(f"    ⚠️  Warning: Diagonal not zero (expected for PDC)")
        else:
            print(f"    ✓ Diagonal is zero (correct)")
        
        # Check available bands
        available_bands = [f for f in sample.files if 'pdc_' in f]
        print(f"    ✓ Available bands: {len(available_bands)}")
        print(f"      {', '.join(available_bands[:5])}{'...' if len(available_bands) > 5 else ''}")
    
    # Count total epochs
    print(f"\n[4] Counting total epochs...")
    
    def count_epochs(files):
        total = 0
        for f in files:
            try:
                data = np.load(f)
                total += len(data['labels'])
            except:
                pass
        return total
    
    epilepsy_epochs = count_epochs(epilepsy_files)
    control_epochs = count_epochs(control_files)
    
    print(f"  Epilepsy: {epilepsy_epochs:,} epochs from {len(epilepsy_files)} files")
    print(f"  Control:  {control_epochs:,} epochs from {len(control_files)} files")
    print(f"  Total:    {epilepsy_epochs + control_epochs:,} epochs")
    
    # Balance check
    ratio = epilepsy_epochs / max(control_epochs, 1)
    print(f"\n[5] Dataset balance:")
    print(f"  Ratio (epilepsy/control): {ratio:.2f}")
    if 0.5 <= ratio <= 2.0:
        print(f"  ✓ Reasonably balanced")
    else:
        print(f"  ⚠️  Imbalanced - consider class weights in training")
    
    # Optional: Check if epoch files exist
    if EPOCHS_DIR.exists():
        print(f"\n[6] Checking epoch files (for additional features)...")
        epoch_files = list(EPOCHS_DIR.rglob("*_epochs.npy"))
        print(f"  ✓ Found {len(epoch_files)} epoch files")
        
        if len(epoch_files) > 0:
            sample_epoch = np.load(epoch_files[0])
            print(f"  ✓ Sample shape: {sample_epoch.shape}")
            print(f"  → Can use for spectral/statistical features")
    else:
        print(f"\n[6] Epoch directory not found: {EPOCHS_DIR}")
        print(f"  → Will use connectivity features only")
    
    print(f"\n{'='*80}")
    print(f"✅ VALIDATION PASSED!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  • {len(epilepsy_files)} epilepsy connectivity files")
    print(f"  • {len(control_files)} control connectivity files")
    print(f"  • {epilepsy_epochs + control_epochs:,} total epochs")
    print(f"  • {N_CHANNELS} EEG channels")
    print(f"  • Pre-computed PDC ready to use")
    
    print(f"\nNext step: Run step2_extract_features.py")
    
    return True


# =============================================================================
# HELPER: Generate file mapping
# =============================================================================

def generate_file_mapping():
    """Create a mapping file for quick reference."""
    
    npz_files = list(CONNECTIVITY_DIR.rglob("*_graphs.npz"))
    
    mapping = {
        'epilepsy': [],
        'control': [],
        'unknown': []
    }
    
    for f in npz_files:
        path_str = str(f)
        rel_path = str(f.relative_to(CONNECTIVITY_DIR))
        
        if EPILEPSY_DIR_PATTERN in path_str:
            mapping['epilepsy'].append(rel_path)
        elif CONTROL_DIR_PATTERN in path_str:
            mapping['control'].append(rel_path)
        else:
            mapping['unknown'].append(rel_path)
    
    # Save mapping
    import json
    with open(OUTPUT_DIR / 'file_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Saved file mapping to: {OUTPUT_DIR / 'file_mapping.json'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    success = validate_dataset()
    
    if success:
        generate_file_mapping()
