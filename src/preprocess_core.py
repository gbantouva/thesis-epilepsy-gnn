# src/preprocess_core.py

# EEG Epilepsy Preprocessing Pipeline

#This repository contains a reproducible EEG preprocessing pipeline for epilepsy detection using the Temple University Hospital (TUH) EEG Epilepsy Corpus.

## Overview

#The pipeline processes raw EEG recordings in EDF format by applying:

#- Channel cleaning and standard 10-20 electrode montage assignment  
#- Average referencing, notch filtering at 60 Hz, and bandpass filtering (0.5–100 Hz)  
#- Optional ICA artifact removal (disabled by default)  
#- Resampling to 250 Hz and optional cropping of start segments for controls  
#- Fixed-length epoching with amplitude-based artifact rejection  
#- Per-epoch, per-channel z-score normalization  
#- Label extraction from filename conventions indicating epileptic vs non-epileptic status  
#- Optional power spectral density (PSD) computation before and after preprocessing  

#Processed epochs, labels, raw data, metadata, and PSD plots are saved for downstream machine learning and graph-based analyses.

## Running the Pipeline

#The core preprocessing logic is in `src/preprocess_core.py`. To preprocess a single EDF file and save outputs (epochs, labels, raw data, PSD plots), run:

#python src/preprocess_single.py --edf path/to/file.edf --out path/to/output_dir --psd_dir path/to/psd_figures

from pathlib import Path
import re, pickle, numpy as np, mne
from mne.preprocessing import ICA

CORE_CHS = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T1","T3","C3","Cz","C4","T4","T2",
            "T5","P3","Pz","P4","T6","O1","Oz","O2"]

def clean_channel_names(raw: mne.io.BaseRaw):
    """
    Clean EEG channel names in raw data by removing prefixes and suffixes to standardize channel labels.

    Parameters:
    -----------
    raw : mne.io.BaseRaw
        Raw EEG data object from MNE.
    
    Effects:
    --------
    Modifies raw in-place by renaming channels.
    """
    mapping = {orig: re.sub(r'^(?:EEG\s*)', '', orig).replace('-LE','').replace('-REF','').strip()
               for orig in raw.ch_names}
    raw.rename_channels(mapping)

def set_montage_with_T1T2(raw: mne.io.BaseRaw):
    """
    Set standard 10-20 montage for raw EEG data and manually add T1, T2 channel locations 
    which are not included by default in MNE standard montage.

    Parameters:
    -----------
    raw : mne.io.BaseRaw
        Raw EEG data object.
    
    Effects:
    --------
    Modifies raw in-place by setting its montage.
    """
    std = mne.channels.make_standard_montage('standard_1020')
    pos = std.get_positions()['ch_pos']
    pos.update({'T1': [-0.040,-0.090,0.120], 'T2': [0.040,-0.090,0.120]})
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame='head'), match_case=False)

def infer_label_from_path(p: Path) -> int:
    """
    Infer epilepsy label from file path.

    Parameters:
    -----------
    p : Path
        Path object pointing to an EDF file.
    
    Returns:
    --------
    int
        Returns 1 if path matches epilepsy folder pattern, else 0 (non-epilepsy).
    """
    s = str(p).replace('\\','/').lower()
    return 1 if "/00_epilepsy/" in s else 0

def preprocess_single(
    edf_path: Path,
    notch: float = 60.0,
    band: tuple = (0.5, 100.0),
    resample_hz: float = 250.0,
    epoch_len: float = 2.0,
    epoch_overlap: float = 0.0,
    reject_percentile: float = 95.0,
    crop_first10_if_control: bool = True,
    ica_components=None,
    return_psd: bool = False
):
    """
    Preprocess a single raw EDF EEG file for epilepsy detection.

    Steps:
    - Load raw EDF data (preload)
    - Clean channel names and select core 22 electrodes
    - Apply standard 10-20 montage, manually add T1 and T2
    - Reference signal using common average referencing
    - Apply notch filter at 60 Hz and bandpass filter (0.5-100 Hz)
    - Optional ICA artifact removal (disabled by default)
    - Resample data to fixed frequency (default 250 Hz)
    - Optionally crop first 10 seconds for control (non-epileptic) recordings
    - Epoch into fixed-length segments with specified overlap
    - Artifact rejection based on peak-to-peak amplitude thresholds from percentile
    - Per-epoch, per-channel z-score normalization
    - Generate per-epoch labels inferred from file path
    - Optionally compute power spectral density (PSD) before and after preprocessing

    Parameters:
    -----------
    edf_path : Path
        File path to the raw EEG EDF file.
    notch : float, optional
        Notch filter frequency in Hz, default 60.0 (powerline noise).
    band : tuple of float, optional
        Bandpass filter low and high cutoff frequencies (Hz), default (0.5, 100.0).
    resample_hz : float, optional
        Sampling frequency to resample the data to, default 250 Hz.
    epoch_len : float, optional
        Epoch length in seconds, default 2.0.
    epoch_overlap : float, optional
        Overlap duration between epochs in seconds, default 0.0.
    reject_percentile : float, optional
        Percentile for peak-to-peak amplitude thresholding during artifact rejection, default 95.
    crop_first10_if_control : bool, optional
        Whether to crop first 10 seconds for non-epilepsy files, default True.
    ica_components : list or None, optional
        List of ICA components to exclude (disabled by default).
    return_psd : bool, optional
        Whether to compute and return PSD estimates before and after preprocessing, default False.

    Returns:
    --------
    dict:
        Dictionary containing:
        - 'raw_after' : MNE Raw object post preprocessing,
        - 'epochs' : MNE Epochs object of cleaned and normalized segments,
        - 'labels' : numpy array of epoch labels,
        - 'threshold_uv' : peak-to-peak microvolt threshold used,
        - 'present_channels' : list of channels retained,
        - 'psd_before' (optional) : PSD object before preprocessing,
        - 'psd_after' (optional) : PSD object after preprocessing.
    """

    # Load raw data
    raw_before = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw_before.pick_types(eeg=True)
    clean_channel_names(raw_before)
    raw = raw_before.copy()

    # Pick core channels and set montage
    present = [ch for ch in CORE_CHS if ch in raw.ch_names]
    raw.pick_channels(present); raw_before.pick_channels(present)
    set_montage_with_T1T2(raw)

    # Common average reference and filtering
    raw.set_eeg_reference('average', projection=False)
    raw.notch_filter(freqs=[notch])
    raw.filter(l_freq=band[0], h_freq=band[1], fir_design='firwin', filter_length='auto')

    # Optional ICA artifact removal
    if ica_components:
        ica = ICA(n_components=min(20, len(raw.ch_names)), method='fastica', random_state=42)
        ica.fit(raw)
        #ica.exclude = sorted(set(ica_components))
        #ica.apply(raw)

    # Resample & optional crop
    raw.resample(resample_hz, npad="auto")
    label = infer_label_from_path(edf_path)
    if crop_first10_if_control and label == 0:
        raw.crop(tmin=10.0)

    # Epoching + amplitude artifact rejection
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_len, overlap=epoch_overlap, preload=True)
    X = epochs.get_data()
    max_ptp_uv = np.ptp(X, axis=2).max(axis=1) * 1e6
    thr_uv = float(np.percentile(max_ptp_uv, reject_percentile))
    epochs_clean = epochs.copy().drop_bad(reject=dict(eeg=thr_uv * 1e-6))

    # Z-score
    #Xc = epochs_clean.get_data()
    #m, s = Xc.mean(), Xc.std() if Xc.std() != 0 else 1.0
    #epochs_clean._data = (Xc - m) / s

    # Per-epoch, per-channel z-score normalization
    Xc = epochs_clean.get_data()                     # shape: (n_epochs, n_channels, n_times)
    m = Xc.mean(axis=2, keepdims=True)              # mean over time (per epoch/ch)
    s = Xc.std(axis=2, keepdims=True)               # std over time (per epoch/ch)
    s[s == 0] = 1.0                                 # avoid division by zero
    epochs_clean._data = (Xc - m) / s

    # Labels per epoch
    y = np.full(len(epochs_clean), label, dtype=int)

    # Collect results
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
