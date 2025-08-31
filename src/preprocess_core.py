# src/preprocess_core.py
from pathlib import Path
import re, pickle, numpy as np, mne
from mne.preprocessing import ICA

CORE_CHS = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T1","T3","C3","Cz","C4","T4","T2",
            "T5","P3","Pz","P4","T6","O1","Oz","O2"]

def clean_channel_names(raw):
    mapping = {orig: re.sub(r'^(?:EEG\s*)', '', orig).replace('-LE','').replace('-REF','').strip()
               for orig in raw.ch_names}
    raw.rename_channels(mapping)

def set_montage_with_T1T2(raw):
    std = mne.channels.make_standard_montage('standard_1020')
    pos = std.get_positions()['ch_pos']
    pos.update({'T1': [-0.060,-0.090,0.120], 'T2': [0.060,-0.090,0.120]})
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head'), match_case=False)

def infer_label_from_path(p: Path) -> int:
    s = str(p).replace('\\','/').lower()
    return 1 if "/00_epilepsy/" in s else 0

def preprocess_single(
    edf_path: Path,
    notch=60.0, band=(0.5,100.0), resample_hz=250.0,
    epoch_len=2.0, epoch_overlap=0.0, reject_percentile=95.0,
    crop_first10_if_control=True, ica_components=None,
    return_psd=False
):
    # Load
    raw_before = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw_before.pick_types(eeg=True)
    clean_channel_names(raw_before)
    raw = raw_before.copy()

    # Channels & montage
    present = [ch for ch in CORE_CHS if ch in raw.ch_names]
    raw.pick_channels(present); raw_before.pick_channels(present)
    set_montage_with_T1T2(raw)

    # Ref + filter
    raw.set_eeg_reference('average', projection=False)
    raw.notch_filter(freqs=[notch])
    raw.filter(l_freq=band[0], h_freq=band[1], fir_design='firwin', filter_length='auto')

    # Optional ICA
    if ica_components:
        ica = ICA(n_components=min(20, len(raw.ch_names)), method='fastica', random_state=42)
        ica.fit(raw)
        ica.exclude = sorted(set(ica_components))
        ica.apply(raw)

    # Resample & optional crop
    raw.resample(resample_hz, npad="auto")
    label = infer_label_from_path(edf_path)
    if crop_first10_if_control and label == 0:
        raw.crop(tmin=10.0)

    # Epoching + rejection
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_len, overlap=epoch_overlap, preload=True)
    X = epochs.get_data()
    max_ptp_uv = np.ptp(X, axis=2).max(axis=1) * 1e6
    thr_uv = float(np.percentile(max_ptp_uv, reject_percentile))
    epochs_clean = epochs.copy().drop_bad(reject=dict(eeg=thr_uv * 1e-6))

    # Z-score
    Xc = epochs_clean.get_data()
    m, s = Xc.mean(), Xc.std() if Xc.std() != 0 else 1.0
    epochs_clean._data = (Xc - m) / s

    # Labels per epoch
    y = np.full(len(epochs_clean), label, dtype=int)

    out = {
        "raw_after": raw,
        "epochs": epochs_clean,
        "labels": y,
        "threshold_uv": thr_uv,
        "present_channels": present
    }
    if return_psd:
        out["psd_before"] = raw_before.compute_psd(fmax=band[1], average='mean')
        out["psd_after"]  = raw.compute_psd(fmax=band[1],  average='mean')
    return out
