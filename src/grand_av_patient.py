import sys
from pathlib import Path
import numpy as np
import argparse

# --- Add 'src' to path if needed ---
# This helps Python find your 'grand_av_single' script,
# assuming 'grand_av_single.py' is in a folder named 'src'.
# Adjust if your structure is different.
src_path = Path(__file__).resolve().parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- Import functions from your previous script ---
try:
    # We assume your previous script is saved as 'grand_av_single.py'
    from grand_av_single import (
        load_info,
        compute_psd_grand_average,
        plot_time_grand_average,
        plot_psd_grand_average,
        plot_topomap_band
    )
except ImportError:
    print(f"Error: Could not import from 'grand_av_single.py'.")
    print(f"Make sure it is in your Python path (e.g., in {src_path})")
    sys.exit(1)


def calculate_patient_grand_average(
    patient_id: str,
    data_pp_dir: Path,
    figures_dir: Path
):
    """
    Finds all epoch files for a single patient, computes their true
    grand average in time and frequency, and saves the plots.
    """
    print(f"--- Calculating Grand Average for Patient: {patient_id} ---")

    # 1. Find all epoch files for this patient
    # We search recursively (**) for any file starting with the patient ID
    # and ending in _epochs.npy
    search_pattern = f"**/{patient_id}*_epochs.npy"
    epoch_files = list(data_pp_dir.rglob(search_pattern))

    if not epoch_files:
        print(f"Error: No '*_epochs.npy' files found for patient ID '{patient_id}' "
              f"in {data_pp_dir}")
        return

    print(f"Found {len(epoch_files)} epoch files.")

    # 2. Load and concatenate all epoch arrays
    all_epochs_list = []
    for f in epoch_files:
        try:
            data = np.load(f)
            # Check if (E, C, T) and not empty
            if data.ndim == 3 and data.shape[0] > 0: 
                all_epochs_list.append(data)
            else:
                print(f"Warning: Skipping {f} (empty or wrong dimensions: {data.shape})")
        except Exception as e:
            print(f"Warning: Could not load {f}. Error: {e}")

    if not all_epochs_list:
        print("Error: No valid epoch data loaded. Aborting.")
        return

    # Combine into one large array: (Total_Patient_Epochs, 22, 500)
    all_patient_epochs = np.concatenate(all_epochs_list, axis=0)
    print(f"Total epochs for patient: {all_patient_epochs.shape[0]}")

    # 3. Load info (chs, fs) from the *first* file
    # We assume all files for a patient have the same info (fs, chs)
    first_file = epoch_files[0]
    info_file_name = first_file.stem.replace("_epochs", "_info") + ".pkl"
    info_file = first_file.with_name(info_file_name)
    
    if not info_file.exists():
        print(f"Error: Corresponding info file not found: {info_file}")
        return
    
    chs, fs = load_info(info_file)
    print(f"Loaded info: {len(chs)} channels, {fs} Hz")

    # 4. Ensure output directory exists
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 5. Calculate Time-domain grand average and plot
    print("Calculating time-domain average...")
    # This is the key step: averaging the giant concatenated array
    ga_patient_time = np.mean(all_patient_epochs, axis=0) # Shape (C, T)
    
    time_plot_path = figures_dir / f"{patient_id}_GA_time.png"
    plot_time_grand_average(ga_patient_time, chs, fs, out_png=time_plot_path)
    print(f"Saved time-domain plot to {time_plot_path}")

    # 6. Calculate Frequency-domain (PSD) grand average and plot
    print("Calculating frequency-domain average...")
    # We also pass the giant array to the PSD function
    mean_psd, f = compute_psd_grand_average(all_patient_epochs, fs) # Shape (C, F)

    psd_plot_path = figures_dir / f"{patient_id}_GA_psd.png"
    plot_psd_grand_average(mean_psd, f, chs, out_png=psd_plot_path)
    print(f"Saved PSD plot to {psd_plot_path}")

    # 7. Plot topomap (e.g., Alpha band)
    print("Calculating topomap...")
    topo_plot_path = figures_dir / f"{patient_id}_GA_alpha_topomap.png"
    plot_topomap_band(
        mean_psd, f, chs, band=(8, 12), # e.g., Alpha band
        out_png=topo_plot_path,
        sfreq=fs
    )
    print(f"Saved alpha topomap to {topo_plot_path}")
    print(f"--- Finished Patient: {patient_id} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate true grand average for a single patient "
                    "by combining all their .npy files."
    )
    parser.add_argument(
        "--id", 
        required=True, 
        help="Patient ID (e.g., 'aaaaaanr')"
    )
    parser.add_argument(
        "--data_dir", 
        required=True, 
        help="Path to the *root* preprocessed data directory (e.g., 'data_pp')"
    )
    parser.add_argument(
        "--fig_dir", 
        required=True, 
        help="Path to save the output figures (e.g., 'figures/grand_av_patient')"
    )
    
    args = parser.parse_args()

    calculate_patient_grand_average(
        patient_id=args.id,
        data_pp_dir=Path(args.data_dir),
        figures_dir=Path(args.fig_dir)
    )

    # --- Example Command to Run ---
    # python grand_av_patient.py --id aaaaaanr --data_dir data_pp --fig_dir figures/grand_av_patients