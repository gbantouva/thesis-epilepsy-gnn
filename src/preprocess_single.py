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
    edf = Path(edf); out = Path(out); psd_dir = Path(psd_dir)
    out.mkdir(parents=True, exist_ok=True); psd_dir.mkdir(parents=True, exist_ok=True)

    res = preprocess_single(edf, return_psd=True)  # use defaults from core
    pid = edf.stem

    # Save arrays/metadata
    np.save(out / f"{pid}_epochs.npy", res["epochs"].get_data())
    np.save(out / f"{pid}_labels.npy", res["labels"])
    np.save(out / f"{pid}_raw.npy",    res["raw_after"].get_data())
    with open(out / f"{pid}_info.pkl", "wb") as f:
        pickle.dump(res["raw_after"].info, f)

    # Save PSD PNGs
    for tag, psd in [("before", res.get("psd_before")), ("after", res.get("psd_after"))]:
        if psd is None: continue
        fig = psd.plot(show=False); fig.suptitle(f"PSD {tag.upper()}"); fig.savefig(psd_dir / f"{pid}_PSD_{tag}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


    print(f"Saved epochs={len(res['epochs'])}, thr={res['threshold_uv']:.1f} µV")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--psd_dir", required=True)
    args = ap.parse_args()
    main(args.edf, args.out, args.psd_dir)
