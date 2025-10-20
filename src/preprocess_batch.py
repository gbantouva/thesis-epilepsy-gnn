import sys
from pathlib import Path
import argparse
import traceback
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from preprocess_core import preprocess_single


def main(input_dir: str, output_dir: str, psd_dir: str, max_patients: int = None, pad_missing: bool = False):
    """
    Batch preprocess EEG EDF files from a root directory recursively.

    Parameters:
    -----------
    input_dir : str
        Root folder containing EEG EDF files and subfolders.
    output_dir : str
        Folder to save processed numpy arrays and metadata, preserving input folder structure.
    psd_dir : str
        Folder to save PSD plot images, preserving input folder structure.
    max_patients : int or None
        If set, limits total unique patients processed by their ID extracted from filename.

    This function:
    -------------
    - Recursively finds *.edf files under input_dir.
    - Extracts patient IDs from filenames.
    - Processes EDF files incrementally until max_patients is reached.
    - Calls the core preprocess_single() for each file.
    - Saves numpy arrays and metadata in relative output folder.
    - Saves PSD quality control images in relative PSD folder.
    - Uses tqdm progress bar for feedback.
    - Handles exceptions per file to continue batch.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    psd_path = Path(psd_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    psd_path.mkdir(parents=True, exist_ok=True)
    
    edf_files = list(input_path.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF files in {input_path}")
    
    processed_patients = set()
    
    with tqdm(total=len(edf_files), desc="Processing EDF files") as pbar:
        for edf_file in edf_files:
            pid_full = edf_file.stem
            # Extract patient id from file name - adjust as needed:
            patient_id = pid_full.split('_')[0]
            
            # 1. Determine the expected output path for the epoch file
            relative_path = edf_file.parent.relative_to(input_path)
            output_subdir = output_path / relative_path
            expected_output_file = output_subdir / f"{pid_full}_epochs.npy"

            # 2. Check if the output file already exists
            if expected_output_file.exists():
                # Check for patient limit logic, if this is a NEW patient ID we hit.
                if patient_id not in processed_patients:
                     processed_patients.add(patient_id)

                # Skip processing the file
                tqdm.write(f"Skipping {pid_full}: Output already exists.")
                pbar.update(1)
                continue # Skip the rest of the loop for this file


            if patient_id not in processed_patients:
                if max_patients is not None and len(processed_patients) >= max_patients:
                    print(f"Reached max patient limit: {max_patients}. Stopping.")
                    break
                processed_patients.add(patient_id)
            
            try:
                print(f"Processing {edf_file} (patient {patient_id})...")
                res = preprocess_single(edf_file, return_psd=True, pad_missing=pad_missing)

                X = res["epochs"].get_data()  # (E, C, T)
                if not np.isfinite(X).all():
                    raise ValueError(f"{edf_file}: non-finite values in epochs")

                
                # Preserve folder hierarchy in output and psd dirs
                #relative_path = edf_file.parent.relative_to(input_path)
                
                #output_subdir = output_path / relative_path
                psd_subdir = psd_path / relative_path
                
                output_subdir.mkdir(parents=True, exist_ok=True)
                psd_subdir.mkdir(parents=True, exist_ok=True)

                #np.save(output_subdir / f"{pid_full}_epochs.npy", res["epochs"].get_data())
                #np.save(output_subdir / f"{pid_full}_labels.npy", res["labels"])
                #np.save(output_subdir / f"{pid_full}_raw.npy", res["raw_after"].get_data())
                #np.save(expected_output_file, res["epochs"].get_data()) # Using expected_output_file here
                np.save(expected_output_file, X)
                np.save(output_subdir / f"{pid_full}_labels.npy", res["labels"])
                np.save(output_subdir / f"{pid_full}_raw.npy", res["raw_after"].get_data())
                np.save(output_subdir / f"{pid_full}_present_mask.npy", res["present_mask"])
                with open(output_subdir / f"{pid_full}_info.pkl", "wb") as f:
                    pickle.dump(res["raw_after"].info, f)
                with open(output_subdir / f"{pid_full}_present_channels.json", "w", encoding="utf-8") as f:
                    json.dump(res["present_channels"], f, ensure_ascii=False, indent=2)

                for tag, psd in [("before", res.get("psd_before")), ("after", res.get("psd_after"))]:
                    if psd is None:
                        continue
                    fig = psd.plot(show=False)
                    fig.suptitle(f"PSD {tag.upper()} {pid_full}")
                    fig.savefig(psd_subdir / f"{pid_full}_PSD_{tag}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)
                
                print(f"Finished {pid_full}: saved {len(res['epochs'])} epochs, threshold={res['threshold_uv']:.1f} µV")
            
            except Exception as e:
                print(f"Error processing {edf_file}: {e}")
                traceback.print_exc()
            
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch preprocess EEG EDF files for epilepsy dataset")
    parser.add_argument("--input_dir", required=True, help="Root EEG EDF folder")
    parser.add_argument("--output_dir", required=True, help="Folder to save preprocessed arrays and metadata")
    parser.add_argument("--psd_dir", required=True, help="Folder to save PSD plots")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit unique patients processed")
    # Simple positive flag; default False
    parser.add_argument("--pad-missing", dest="pad_missing", action="store_true",
                        help="Enable zero-padding of missing channels (fixed topology).")
    parser.set_defaults(pad_missing=False)
    
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    
    main(args.input_dir, args.output_dir, args.psd_dir, max_patients=args.max_patients, pad_missing=args.pad_missing)
