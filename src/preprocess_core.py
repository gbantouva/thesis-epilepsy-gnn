from pathlib import Path
import re, pickle, numpy as np, mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
#python src\preprocess_batch.py --input_dir data_raw\DATA --output_dir data_pp --psd_dir figures\psd --max_patients 10 --pad-missing 
# Fixed 10–20 core channel layout (target topology for graphs/ML)
CORE_CHS = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T1","T3","C3","Cz","C4","T4","T2",
            "T5","P3","Pz","P4","T6","O1","Oz","O2"]

def clean_channel_names(raw: mne.io.BaseRaw):
    """
    Standardize raw channel names (remove 'EEG ', '-LE', '-REF', trim whitespace,
    and standardize case to match CORE_CHS). Operates in-place on `raw`.
    """
    # 1. General cleaning
    mapping = {orig: re.sub(r'^(?:EEG\s*)', '', orig).replace('-LE','').replace('-REF','').strip()
               for orig in raw.ch_names}
    raw.rename_channels(mapping)
    
    # 2. Fix capitalization (e.g., "FP1" -> "Fp1")
    # Build a map of {lowercase_name: correct_case_name}
    core_ch_map = {ch.lower(): ch for ch in CORE_CHS}
    
    case_mapping = {}
    for ch in raw.ch_names:
        if ch.lower() in core_ch_map:
            case_mapping[ch] = core_ch_map[ch.lower()]
            
    raw.rename_channels(case_mapping)

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
    band: tuple = (1.0, 100.0),  # Tuned for ICLabel
    resample_hz: float = 250.0,
    epoch_len: float = 2.0,
    epoch_overlap: float = 0.0,
    reject_percentile: float = 95.0,
    crop_first10_if_control: bool = True,
    ica_components: bool = False,
    return_psd: bool = False,
    pad_missing: bool = True,
    ica_dir: str = None
):
    """
    End-to-end preprocessing for a single EDF (v13 - Final Order):
      - Load, clean names (with case-correction)
      - Pick *only* present CORE_CHS channels (no padding)
      - Filter (1-100 Hz)
      - Set Montage
      - Common Average Reference
      - Automated ICA artifact removal (ICLabel + Infomax)
      - Pad missing channels (to get 22)
      - Resample, crop, epoch, reject (w/ Sanity Cap), z-score
    """
    edf_path = Path(edf_path)
    pid = edf_path.stem
    
    # Load raw data
    raw_before = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw_before.pick_types(eeg=True, exclude=['EOG', 'ECG', 'EMG', 'MISC', 'STIM']) # Try to exclude junk
    clean_channel_names(raw_before) # Fixes case
    raw = raw_before.copy()

    # --- 1. PICK (NO PADDING) ---
    # We select *only* the channels we care about, throwing away 'EKG', 'DC', etc.
    # We also run this on raw_before for the PSD plot.
    pick_order_and_pad(raw_before, pad_missing=False) 
    present, present_mask_interim = pick_order_and_pad(raw, pad_missing=False)
    
    if len(present) == 0:
        raise RuntimeError("No CORE_CHS channels were found in this file.")

    # --- 2. Filtering ---
    raw.notch_filter(freqs=[notch])
    raw.filter(l_freq=band[0], h_freq=band[1], fir_design='firwin', filter_length='auto')
    
    # --- 3. Set Montage ---
    # This will now work, as 'raw' only contains known, case-corrected EEG channels
    set_montage_for_corechs(raw)

    # --- 4. Re-reference (CAR) ---
    raw.set_eeg_reference('average', projection=False)

    # --- 5. ICA Artifact Removal ---
    if ica_components:
        try:
            n_ica_comp = min(20, len(raw.ch_names) - 1)
            
            if n_ica_comp < 2:
                print(f"   ! Skipping ICA: Not enough channels ({len(raw.ch_names)})")
            else:
                ica = ICA(
                    n_components=n_ica_comp, 
                    method='infomax', 
                    fit_params=dict(extended=True),
                    random_state=42
                )
                ica.fit(raw)

                ic_labels = label_components(raw, ica, method="iclabel")
                labels = ic_labels["labels"]
                ica.exclude = [i for i, label in enumerate(labels) if label not in ["brain", "other"]]

                if ica.exclude:
                    print(f"   ICA: Found {len(ica.exclude)} artifact components. Plotting...")
                    if ica_dir:
                        ica_plot_dir = Path(ica_dir)
                        ica_plot_dir.mkdir(parents=True, exist_ok=True)
                        for comp_idx in ica.exclude:
                            fig = ica.plot_properties(raw, picks=comp_idx, show=False)[0]
                            fig.savefig(ica_plot_dir / f"{pid}_ica_comp_{comp_idx:02d}_REMOVED.png", dpi=100)
                            plt.close(fig)
                            
                    print(f"   ICA: Removing {len(ica.exclude)} components.")
                    ica.apply(raw)
                else:
                    print("   ICA: No artifacts found to remove.")
                
        except Exception as e:
            print(f"   ! ICA failed for {edf_path.stem}: {e}")
            
    # --- 6. PAD (NOW a_components=False,
    # Now we pad with zeros to get our fixed 22-channel graph
    # We also get the *final* present_mask
    present_final, present_mask = pick_order_and_pad(raw, pad_missing=True)

    # --- 7. Resample & optional crop ---
    raw.resample(resample_hz, npad="auto")
    label = infer_label_from_path(edf_path)
    if crop_first10_if_control and label == 0:
        raw.crop(tmin=10.0)

    # --- 8. Epoching + amplitude artifact rejection (Sanity Cap) ---
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_len, overlap=epoch_overlap, preload=True)
    X = epochs.get_data() # This gets data in VOLTS
    if X.shape[0] == 0:
        print(f"   ! No epochs created for {edf_path.stem}. Skipping file.")
        return None
    
    max_ptp_uv = np.ptp(X, axis=2).max(axis=1) * 1e6
    
    adaptive_thr_uv = float(np.percentile(max_ptp_uv, reject_percentile))
    sanity_thr_uv = 500.0  # Hard cap
    final_thr_uv = min(adaptive_thr_uv, sanity_thr_uv)
    
    print(f"   Adaptive threshold (95th percentile): {adaptive_thr_uv:.1f} uV")
    print(f"   Sanity-capped threshold: {final_thr_uv:.1f} uV (Using this one)")
    
    epochs_clean = epochs.copy().drop_bad(reject=dict(eeg=final_thr_uv * 1e-6))

    if len(epochs_clean) == 0:
        print(f"   ! All epochs rejected for {edf_path.stem} (Threshold={final_thr_uv:.1f} uV). Skipping file.")
        return None
    
    # --- 9. Per-epoch, per-channel z-score across time ---
    Xc = epochs_clean.get_data()
    m = Xc.mean(axis=2, keepdims=True)
    s = Xc.std(axis=2, keepdims=True)
    s[s == 0] = 1.0
    Xz = (Xc - m) / s
    epochs_clean = mne.EpochsArray(
        Xz, epochs_clean.info, events=epochs_clean.events,
        tmin=epochs_clean.tmin, event_id=epochs_clean.event_id, on_missing='ignore'
    )

    # --- 10. Labels per epoch (subject-level from path) ---
    y = np.full(len(epochs_clean), label, dtype=int)

    # --- Collect results ---
    out = {
        "raw_after": raw,
        "epochs": epochs_clean,
        "labels": y,
        "threshold_uv": final_thr_uv,
        "present_channels": present_final, # The final list of 22
        "present_mask": present_mask,   # The final 22-long mask
    }

    if return_psd:
        # We need to pad raw_before too so its PSD matches
        pick_order_and_pad(raw_before, pad_missing=True)
        out["psd_before"] = raw_before.compute_psd(fmax=band[1], average='mean')

        ra = raw.copy()
        if present:
            ra.pick_channels(present) # Plot only the *real* channels
        out["psd_after"] = ra.compute_psd(fmax=band[1], average='mean')
        
    return out