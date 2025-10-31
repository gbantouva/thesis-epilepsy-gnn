from pathlib import Path
import re, pickle, numpy as np, mne
from mne.preprocessing import ICA
#python src\preprocess_batch.py --input_dir data_raw\DATA --output_dir data_pp --psd_dir figures\psd --max_patients 10 --pad-missing 
# Fixed 10–20 core channel layout (target topology for graphs/ML)
CORE_CHS = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T1","T3","C3","Cz","C4","T4","T2",
            "T5","P3","Pz","P4","T6","O1","Oz","O2"]

def clean_channel_names(raw: mne.io.BaseRaw):
    """
    Standardize raw channel names (remove 'EEG ', '-LE', '-REF', trim whitespace).
    Operates in-place on `raw`.
    """
    mapping = {orig: re.sub(r'^(?:EEG\s*)', '', orig).replace('-LE','').replace('-REF','').strip()
               for orig in raw.ch_names}
    raw.rename_channels(mapping)

def pick_order_and_pad(raw: mne.io.BaseRaw, pad_missing: bool = True):
    """
    Reorder channels to match CORE_CHS; optionally zero-pad missing ones.
    Returns:
      - ordered_list: list of channel names after reordering (CORE_CHS if padded)
      - present_mask: bool[22] True where original channel existed, False if padded
    Raises when no CORE_CHS are present and pad_missing=False.
    """
    present = [ch for ch in CORE_CHS if ch in raw.ch_names]  # in CORE_CHS order
    if not present and not pad_missing:
        raise RuntimeError("No recognizable CORE_CHS channels found in this recording.")

    if pad_missing:
        missing = [ch for ch in CORE_CHS if ch not in raw.ch_names]
        if missing:
            data = np.zeros((len(missing), raw.n_times))
            info = mne.create_info(missing, sfreq=raw.info["sfreq"], ch_types="eeg")
            raw.add_channels([mne.io.RawArray(data, info)], force_update_info=True)
        raw.reorder_channels(CORE_CHS)
        present_mask = np.array([ch in present for ch in CORE_CHS], dtype=bool)
        return CORE_CHS, present_mask

    # keep only the present subset, ordered
    raw.pick_channels(present)
    raw.reorder_channels(present)
    present_mask = np.array([ch in present for ch in CORE_CHS], dtype=bool)
    return present, present_mask

def set_montage_for_corechs(raw: mne.io.BaseRaw):
    """
    Assign 10–20 electrode positions for channels present in `raw`.
    Uses standard_1020 where available; approximates T1/T2 (FT9/FT10-like coords).
    """
    std = mne.channels.make_standard_montage('standard_1020')
    pos = std.get_positions()['ch_pos'].copy()
    ch_pos = {}
    # Add coordinates for channels that exist in this recording
    for ch in raw.ch_names:
        if ch in pos:
            ch_pos[ch] = pos[ch]
    # mastoids (approximate)
    # Add approximate mastoid positions for T1/T2 if present
    # FT9 : [-0.0840759 0.0145673 -0.050429 ] FT10 : [ 0.0841131 0.0143647 -0.050538 ]
    if 'T1' in raw.ch_names:
        ch_pos.setdefault('T1', (-0.0840759, 0.0145673, -0.050429))
    if 'T2' in raw.ch_names:
        ch_pos.setdefault('T2', (0.0841131, 0.0143647, -0.050538))
    #pos.update({'T1': [-0.040,-0.090,0.120], 'T2': [0.040,-0.090,0.120]})
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, match_case=False)

def infer_label_from_path(p: Path) -> int:
    """
    Infer binary label from path:
      returns 1 if path contains '/00_epilepsy/', else 0 (control).
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
    return_psd: bool = False,
    pad_missing: bool = True               # << default: pad with zeros for fixed topology
):
    """
    End-to-end preprocessing for a single EDF:
      - Load EDF, keep EEG, clean names
      - Enforce CORE_CHS order; optionally pad missing channels
      - Set montage (incl. T1/T2 approximations)
      - Common average reference, notch, band-pass
      - Optional ICA (fitting scaffold present)
      - Resample; optionally crop first 10s for controls
      - Epoch, percentile-based amplitude rejection
      - Per-epoch, per-channel z-scoring across time
      - Create per-epoch labels from path
      - (Optional) PSD before/after
    Returns dict with processed Raw, Epochs, labels, thresholds, channel info, and masks.
    """
    edf_path = Path(edf_path)
    # Load raw data
    raw_before = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw_before.pick_types(eeg=True)
    clean_channel_names(raw_before)
    raw = raw_before.copy()

    # --- Enforce your channel order; pad missing if requested ---
    if pad_missing and not any(ch in raw_before.ch_names for ch in CORE_CHS):
        raise RuntimeError("No CORE_CHS present in this recording; aborting to avoid all-zero data.")
    present, present_mask = pick_order_and_pad(raw, pad_missing=pad_missing)
    pick_order_and_pad(raw_before, pad_missing=pad_missing)  # mirror for PSD-before alignment

    # --- Montage consistent with CORE_CHS names ---
    set_montage_for_corechs(raw)

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

    # --- Per-epoch, per-channel z-score across time ---
    Xc = epochs_clean.get_data()
    m = Xc.mean(axis=2, keepdims=True)
    s = Xc.std(axis=2, keepdims=True)
    s[s == 0] = 1.0
    Xz = (Xc - m) / s
    epochs_clean = mne.EpochsArray(
        Xz, epochs_clean.info, events=epochs_clean.events,
        tmin=epochs_clean.tmin, event_id=epochs_clean.event_id, on_missing='ignore'
    )

    # --- Labels per epoch (subject-level from path) ---
    y = np.full(len(epochs_clean), label, dtype=int)


    # Collect results
    out = {
        "raw_after": raw,
        "epochs": epochs_clean,
        "labels": y,
        "threshold_uv": thr_uv,
        "present_channels": present,
        "present_mask": present_mask,  # True for originally present CORE_CHS

    }
    if return_psd:
        # channels that actually existed in the EDF
        real_chs = [ch for ch, m in zip(CORE_CHS, present_mask) if m]

        # BEFORE: compute PSD on real channels only (avoid padded zeros)
        rb = raw_before.copy()
        if real_chs:                      # only pick if we have any real channels
            rb.pick_channels(real_chs)
        out["psd_before"] = rb.compute_psd(fmax=band[1], average='mean')

        # AFTER: (recommended) also drop padded channels for a clean plot
        ra = raw.copy()
        if real_chs:
            ra.pick_channels(real_chs)
        out["psd_after"] = ra.compute_psd(fmax=band[1], average='mean')
        #out["psd_before"] = raw_before.compute_psd(fmax=band[1], average='mean')
        #out["psd_after"]  = raw.compute_psd(fmax=band[1],  average='mean')
    return out
