import numpy as np
from pathlib import Path
import os
import sys

# Add src to path just in case we need core functions (e.g., infer_label_from_path)
sys.path.append(str(Path(__file__).resolve().parent))
from preprocess_core import infer_label_from_path

# --- CONFIGURATION ---
INPUT_EPOCHS_DIR = Path("data_pp")
OUTPUT_AVERAGE_DIR = Path("figures/grand_average")
OUTPUT_AVERAGE_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------

def compute_grand_average():
    """
    Loads all clean epoch arrays, computes the Subject Average (mean epoch),
    groups them by label (Epilepsy/Control), and computes the final Grand Average
    for each group.
    """
    print(f"Loading epoch files from: {INPUT_EPOCHS_DIR}")

    # Find all preprocessed epoch files
    epoch_files = list(INPUT_EPOCHS_DIR.rglob('*_epochs.npy'))
    
    if not epoch_files:
        print("ERROR: No _epochs.npy files found. Did you run preprocess_batch.py?")
        return

    # Dictionary to store the mean epoch (Subject Average) for each group
    subject_averages = {
        1: [],  # Epilepsy (label 1)
        0: []   # Control (label 0)
    }
    
    # 1. Subject Average Calculation (First Level)
    for file_path in epoch_files:
        try:
            # The label is inferred from the original EDF path, but since we don't have 
            # the original path here, we can infer it from the output file's name or a stored label file.
            # Easiest way: Assume label 1 files were created by patients in the '00_epilepsy' folder structure.
            # In your batch script, you can save a simple text file with the label, but 
            # for now, we'll try to guess based on the ID's structure if possible, 
            # or rely on the saved labels array if needed.
            
            # --- USING INFERRED LABEL LOGIC (Slight modification needed here): ---
            # NOTE: For maximum accuracy, you should load the original EDF path 
            # or the corresponding [pid]_labels.npy, but for simplicity, 
            # we will assume you load the label from the saved file:
            
            pid = file_path.stem.replace('_epochs', '')
            label_file = file_path.parent / f"{pid}_labels.npy"
            
            # Load the first label, assuming all epochs in the file have the same label
            label_array = np.load(label_file)
            label = int(label_array[0])
            
            epochs = np.load(file_path)  # Shape: (N_epochs, N_channels, N_times)
            
            if epochs.size == 0:
                print(f"Warning: Skipping empty file {file_path.name}")
                continue

            # Subject Average: Mean across the epoch dimension (axis=0)
            subject_avg = np.mean(epochs, axis=0) # Shape: (N_channels, N_times)
            
            subject_averages[label].append(subject_avg)
            
        except Exception as e:
            print(f"Could not process {file_path.name}: {e}")
            
    # 2. Grand Average Calculation (Second Level)
    for label, averages in subject_averages.items():
        group_name = "epilepsy" if label == 1 else "control"
        
        if not averages:
            print(f"No data found for the {group_name} group.")
            continue
            
        # Stack all subject averages (N_subjects x N_channels x N_times)
        stacked_averages = np.stack(averages, axis=0)
        
        # Grand Average: Mean across the subject dimension (axis=0)
        grand_average = np.mean(stacked_averages, axis=0)
        
        # 3. Save the result
        output_filename = OUTPUT_AVERAGE_DIR / f"grand_average_{group_name}.npy"
        np.save(output_filename, grand_average)
        
        print(f"Successfully calculated and saved Grand Average for {group_name.upper()} group.")
        print(f"Result shape: {grand_average.shape}") # Should be (N_channels, N_times)

if __name__ == "__main__":
    compute_grand_average()