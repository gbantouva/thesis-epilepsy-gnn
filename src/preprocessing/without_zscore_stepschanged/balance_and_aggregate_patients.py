import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
from tqdm import tqdm

def balance_and_aggregate_patients(data_dir: Path, output_dir: Path,
                                   min_epochs: int = None):
    """
    Create balanced patient-level signatures with PRESERVED DIRECTORY HIERARCHY.

    (MEMORY-SAFE VERSION with hierarchy preservation)
    
    For each patient:
    1. Find all their epoch files and count totals.
    2. Determine minimum N epochs.
    3. Randomly select N indices from the total epochs.
    4. Load *only* the selected epochs from disk.
    5. Save balanced patient data in SAME relative path structure.
    
    Args:
        data_dir: Preprocessed data directory
        output_dir: Where to save balanced data (will mirror input structure)
        min_epochs: Minimum epochs to sample (None = auto-detect)
    """
    
    print("="*70)
    print("PATIENT-LEVEL EPOCH BALANCING (Memory-Safe + Hierarchy Preserved)")
    print("="*70)
    
    # Step 1: Discover all patients and count epochs
    print("Step 1: Discovering patients and counting epochs...")
    patients = defaultdict(lambda: {
        "label": None, 
        "files": [], 
        "total_epochs": 0, 
        "file_epoch_counts": [],
        "relative_paths": []  # NEW: Store relative paths
    })
    
    all_epoch_files = sorted(list(data_dir.rglob("*_epochs.npy")))
    
    for epoch_file in tqdm(all_epoch_files, desc="Discovering files"):
        # Determine label
        path_str = str(epoch_file).replace('\\', '/').lower()
        if '/00_epilepsy/' in path_str or '/epilepsy/' in path_str:
            label = 1
        elif '/01_no_epilepsy/' in path_str or '/control/' in path_str:
            label = 0
        else:
            labels_file = epoch_file.parent / f"{epoch_file.stem.replace('_epochs', '')}_labels.npy"
            if labels_file.exists():
                label = int(np.load(labels_file)[0])
            else:
                continue
        
        # Extract patient ID from directory structure
        try:
            rel_path = epoch_file.relative_to(data_dir)
            # Assuming structure: 00_epilepsy/patient_id/session/montage/file.npy
            # Extract patient_id (second level in hierarchy)
            patient_id = rel_path.parts[1] if len(rel_path.parts) > 1 else rel_path.parts[0]
        except:
            patient_id = epoch_file.stem.split('_')[0]
        
        # Count epochs
        data = np.load(epoch_file, mmap_mode='r')  # mmap_mode is memory-safe
        n_epochs = data.shape[0]
        
        patients[patient_id]["label"] = label
        patients[patient_id]["files"].append(epoch_file)
        patients[patient_id]["total_epochs"] += n_epochs
        patients[patient_id]["file_epoch_counts"].append(n_epochs)
        
        # Store relative path for later
        try:
            patients[patient_id]["relative_paths"].append(epoch_file.relative_to(data_dir))
        except:
            patients[patient_id]["relative_paths"].append(None)
        
    print(f"Found {len(patients)} patients")
    
    # Step 2: Determine minimum epoch count
    epoch_counts = [p["total_epochs"] for p in patients.values()]
    
    if min_epochs is None:
        min_epochs = min(epoch_counts)
    
    print(f"\nEpoch distribution:")
    print(f"  Min: {min(epoch_counts)}")
    print(f"  Max: {max(epoch_counts)}")
    print(f"  Mean: {np.mean(epoch_counts):.1f}")
    print(f"  Median: {np.median(epoch_counts):.1f}")
    print(f"\nUsing {min_epochs} epochs per patient (minimum)")
    
    # Step 3: Sample and save balanced data (MEMORY-SAFE + HIERARCHICAL)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for patient_id, patient_data in tqdm(patients.items(), desc="Balancing patients"):
        
        # Select N random indices from the *total* count
        total_epochs = patient_data["total_epochs"]
        
        # Handle cases with fewer epochs by sampling with replacement
        replace = total_epochs < min_epochs
        if replace:
            print(f"\n  ⚠️ {patient_id}: Only {total_epochs} epochs. Sampling with replacement to get {min_epochs}.")
        
        chosen_indices = np.random.choice(total_epochs, min_epochs, replace=replace)
        chosen_indices.sort()
        
        balanced_epochs_list = []
        indices_to_get = list(chosen_indices)
        current_epoch_idx = 0
        
        # Track which file contributed the most epochs (for output path)
        file_epoch_contributions = []
        
        # Loop through files, loading only what's needed
        for i, epoch_file in enumerate(patient_data["files"]):
            n_epochs_in_file = patient_data["file_epoch_counts"][i]
            
            # Find which of our chosen indices fall within this file
            file_epoch_indices = [
                idx - current_epoch_idx for idx in indices_to_get
                if current_epoch_idx <= idx < (current_epoch_idx + n_epochs_in_file)
            ]
            
            if file_epoch_indices:
                # Load the file *now* and get only the indices we need
                data = np.load(epoch_file)
                balanced_epochs_list.append(data[file_epoch_indices])
                
                # Track contribution
                file_epoch_contributions.append((len(file_epoch_indices), i))
                
                # Remove these indices from our "to-get" list
                indices_to_get = [idx for idx in indices_to_get if idx >= (current_epoch_idx + n_epochs_in_file)]
            
            current_epoch_idx += n_epochs_in_file
            
            if not indices_to_get:
                break  # We found all our indices
        
        # Concatenate the chunks we loaded
        balanced_epochs = np.concatenate(balanced_epochs_list, axis=0)
        
        # === NEW: Determine output path preserving hierarchy ===
        # Use the file that contributed the most epochs as the "representative" path
        file_epoch_contributions.sort(reverse=True)  # Sort by contribution
        representative_file_idx = file_epoch_contributions[0][1]
        representative_file = patient_data["files"][representative_file_idx]
        representative_rel_path = patient_data["relative_paths"][representative_file_idx]
        
        if representative_rel_path is not None:
            # Create the same directory structure in output
            output_subdir = output_dir / representative_rel_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Base filename (remove _epochs.npy)
            base_filename = representative_rel_path.stem.replace("_epochs", "")
            
            # Save balanced data in hierarchical structure
            output_epochs_path = output_subdir / f"{base_filename}_epochs_balanced.npy"
            output_info_path = output_subdir / f"{base_filename}_info.pkl"
            output_labels_path = output_subdir / f"{base_filename}_labels_balanced.npy"
        else:
            # Fallback: flat structure
            output_subdir = output_dir
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_epochs_path = output_subdir / f"{patient_id}_epochs_balanced.npy"
            output_info_path = output_subdir / f"{patient_id}_info.pkl"
            output_labels_path = output_subdir / f"{patient_id}_labels_balanced.npy"
        
        # Save balanced epochs
        np.save(output_epochs_path, balanced_epochs)
        
        # Copy metadata from representative file
        sample_file = representative_file
        pid = sample_file.stem.replace("_epochs", "")
        
        # Copy info
        info_file = sample_file.parent / f"{pid}_info.pkl"
        if info_file.exists():
            with open(info_file, 'rb') as f:
                info = pickle.load(f)
            with open(output_info_path, 'wb') as f:
                pickle.dump(info, f)
        
        # Save label
        labels = np.full(min_epochs, patient_data["label"])
        np.save(output_labels_path, labels)
    
    print(f"\n✓ Balanced data saved to: {output_dir}")
    print(f"  Each patient now has exactly {min_epochs} epochs")
    print(f"  Directory hierarchy preserved from input structure")


if __name__ == "__main__":
    balance_and_aggregate_patients(
        data_dir=Path(r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp\without_zscore_stepschanged"),
        output_dir=Path(r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp\\without_zscore_stepschanged\data_pp_balanced"),
        min_epochs=None  # Auto-detect minimum
    )