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
import json


from preprocess_core import preprocess_single

def main(edf:str, out:str, psd_dir:str, pad_missing: bool):
    """
    Preprocess a single raw EEG EDF file and save preprocessed outputs.

    Saves:
      - {pid}_epochs.npy           : epochs array (n_epochs, n_ch, n_times)
      - {pid}_labels.npy           : per-epoch labels
      - {pid}_raw.npy              : raw AFTER preprocessing (channels x times)
      - {pid}_info.pkl             : MNE Raw.info for the post-processed raw
      - {pid}_present_mask.npy     : bool[22] mask (True if real channel, False if padded)
      - {pid}_present_channels.json: ordered channel names after pick/pad
      - {pid}_PSD_before.png/.png  : PSD figs (if available)
    """
    edf = Path(edf); out = Path(out); psd_dir = Path(psd_dir)
    out.mkdir(parents=True, exist_ok=True); psd_dir.mkdir(parents=True, exist_ok=True)
    
    # Run preprocessing pipeline on single EDF file
    #res = preprocess_single(edf, return_psd=True, pad_missing=args.pad_missing)  # use defaults from core
    res = preprocess_single(edf, return_psd=True, pad_missing=pad_missing)  # use defaults from core
    pid = edf.stem

    # Save preprocessed numpy arrays and metadata
    np.save(out / f"{pid}_epochs.npy", res["epochs"].get_data())
    np.save(out / f"{pid}_labels.npy", res["labels"])
    np.save(out / f"{pid}_raw.npy",    res["raw_after"].get_data())
    np.save(out / f"{pid}_present_mask.npy", res["present_mask"])
    with open(out / f"{pid}_info.pkl", "wb") as f:
        pickle.dump(res["raw_after"].info, f)
    with open(out / f"{pid}_present_channels.json", "w", encoding="utf-8") as f:
        json.dump(res["present_channels"], f, ensure_ascii=False, indent=2)
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
    parser.add_argument("--pad_missing", action="store_true", help="Zero-pad missing CORE_CHS")
    args = parser.parse_args()
    #main(args.edf, args.out, args.psd_dir)
    main(args.edf, args.out, args.psd_dir, args.pad_missing)