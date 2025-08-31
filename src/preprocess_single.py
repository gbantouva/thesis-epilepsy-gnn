"""
Preprocess a single EDF EEG file from TUEP dataset.
Usage:
    python src/preprocess_single.py --edf data_raw/DATA/00_epilepsy/...edf --out data_pp --psd figures/psd
"""

import os
import re
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from mne.preprocessing import ICA

# --------------------------
# Config
# --------------------------
CORE_CHS = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'T1','T3','C3','Cz','C4','T4','T2',
    'T5','P3','Pz','P4','T6','O1','Oz','O2'
]

EPOCH_LEN = 2.0
EPOCH_OVERLAP = 0.0
RESAMPLE_HZ = 250

# --------------------------
# Helpers
# --------------------------

def clean_channel_names(raw):
    """Remove prefixes/suffixes (EEG F3-REF → F3)."""
    mapping = {
        orig: re.sub(r'^(?:EEG\s*)', '', orig).replace('-LE','').replace('-REF','').strip()
        for orig in raw.ch_names
    }
    raw.rename_channels(mapping)
    return raw

def preprocess_single_edf(edf_path, out_dir, psd_dir):
    """Preprocess a single EDF file."""
    edf_path = Path(edf_path)
    out_dir = Path(out_dir)
    psd_dir = Path(psd_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    psd_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
    raw.pick_types(eeg=True)
    raw = clean_channel_names(raw)

    # ---- Pick core channels ----
    present = [ch for ch in CORE_CHS if ch in raw.ch_names]
    raw.pick_channels(present)

    # ---- Montage ----
    std_montage = mne.channels.make_standard_montage('standard_1020')
    pos = std_montage.get_positions()['ch_pos']
    pos.update({'T1': [-0.060,-0.090,0.120], 'T2': [0.060,-0.090,0.120]})
    montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head')
    raw.set_montage(montage, match_case=False)

    # ---- Reference & filtering ----
    raw.set_eeg_reference('average', projection=False)
    raw.notch_filter(freqs=60)
    raw.filter(l_freq=0.5, h_freq=100)

    # ---- ICA ----
    ica = ICA(n_components=min(20, len(raw.ch_names)), method='fastica', random_state=42)
    ica.fit(raw)
    raw = ica.apply(raw)

    # ---- PSD before/after ----
    patient_id = edf_path.stem
    psd = raw.compute_psd(fmax=100, average='mean')
    fig = psd.plot()
    fig.suptitle(f"PSD After Preprocessing ({patient_id})")
    fig.savefig(psd_dir / f"{patient_id}_PSD.png")
    plt.close(fig)

    # ---- Resample ----
    raw.resample(RESAMPLE_HZ)

    # ---- Crop first 10s for non-epilepsy ----
    if "01_no_epilepsy" in str(edf_path):
        raw.crop(tmin=10.0)

    # ---- Epoching ----
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_LEN, overlap=EPOCH_OVERLAP, preload=True)

    # ---- Artifact rejection ----
    data = epochs.get_data()
    ptp = np.ptp(data, axis=2).max(axis=1) * 1e6
    threshold = np.percentile(ptp, 95)
    reject = dict(eeg=threshold * 1e-6)
    epochs_clean = epochs.copy().drop_bad(reject=reject)

    # ---- Normalize ----
    d = epochs_clean.get_data()
    epochs_clean._data = (d - d.mean()) / d.std()

    # ---- Labels ----
    label = 1 if "00_epilepsy" in str(edf_path) else 0
    labels = np.full(len(epochs_clean), label, dtype=int)

    # ---- Save ----
    prefix = out_dir / patient_id
    np.save(f"{prefix}_epochs.npy", epochs_clean.get_data())
    np.save(f"{prefix}_labels.npy", labels)
    np.save(f"{prefix}_raw.npy", raw.get_data())
    with open(f"{prefix}_info.pkl", "wb") as f:
        pickle.dump(raw.info, f)

    print(f"✅ Saved {len(epochs_clean)} epochs for {patient_id} to {out_dir}")


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True, help="Path to EDF file")
    ap.add_argument("--out", required=True, help="Output directory for npy/pkl")
    ap.add_argument("--psd", required=True, help="Directory for PSD plots")
    args = ap.parse_args()

    preprocess_single_edf(args.edf, args.out, args.psd)
