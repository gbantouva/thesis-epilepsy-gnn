"""
Usage:
  python src/preprocess_single.py --edf path/to/file.edf --out path/to/output_dir --psd_dir path/to/psd_figures
  python src/preprocess_single.py --edf data_raw/DATA/...t001.edf --out data_pp --psd_dir figures/psd
"""
import sys
from pathlib import Path

# Let this script import sibling modules from src/
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from preprocess_core import preprocess_single

def main(edf:str, out:str, psd_dir:str):
    """
    Preprocess a single raw EEG EDF file and save preprocessed outputs.
    
    Parameters:
    -----------
    edf : str
        Path to the raw EDF file.
    out : str
        Directory path to save numpy arrays and metadata.
    psd_dir : str
        Directory path to save PSD plot images.
    
    This function:
    - Calls the core preprocess_single() function from preprocess_core.py
    - Saves epochs, labels, raw signals as numpy .npy files
    - Saves raw info metadata as pickle
    - Saves PSD plots before and after preprocessing for quality control
    - Prints summary of the saved epochs and threshold
    """
    edf = Path(edf); out = Path(out); psd_dir = Path(psd_dir)
    out.mkdir(parents=True, exist_ok=True); psd_dir.mkdir(parents=True, exist_ok=True)
    
    # Run preprocessing pipeline on single EDF file
    res = preprocess_single(edf, return_psd=True)  # use defaults from core
    pid = edf.stem

    # Save preprocessed numpy arrays and metadata
    np.save(out / f"{pid}_epochs.npy", res["epochs"].get_data())
    np.save(out / f"{pid}_labels.npy", res["labels"])
    np.save(out / f"{pid}_raw.npy",    res["raw_after"].get_data())
    with open(out / f"{pid}_info.pkl", "wb") as f:
        pickle.dump(res["raw_after"].info, f)

    # Save PSD plots before and after preprocessing
    for tag, psd in [("before", res.get("psd_before")), ("after", res.get("psd_after"))]:
        if psd is None: continue
        fig = psd.plot(show=False); fig.suptitle(f"PSD {tag.upper()}"); fig.savefig(psd_dir / f"{pid}_PSD_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


    print(f"Saved epochs={len(res['epochs'])}, thr={res['threshold_uv']:.1f} µV")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a single EEG EDF file for epilepsy detection")
    parser.add_argument("--edf", required=True, help="Path to raw EDF file")
    parser.add_argument("--out", required=True, help="Output directory for arrays and metadata")
    parser.add_argument("--psd_dir", required=True, help="Output directory for PSD figures")
    args = parser.parse_args()
    main(args.edf, args.out, args.psd_dir)
