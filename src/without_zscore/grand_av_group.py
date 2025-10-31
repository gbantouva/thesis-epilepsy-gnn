# src/grand_av_group.py
"""
Group grand averages (Epilepsy vs Control) from preprocessed outputs.

- Walks data_pp recursively, finds * _epochs.npy and the companion files:
    *_present_mask.npy, *_info.pkl, (optional) *_present_channels.json
- Per subject:
    • time-domain mean across epochs -> (C, T)  [NaN for padded chans]
    • PSD per epoch/channel (Welch) -> mean -> (C, F)  [NaN for padded chans]
- Groups by label inferred from path: "/00_epilepsy/" -> 1, else 0
- Saves group grand-averages + figures:
    • group_time_{control,epilepsy}.npy      shape (22, T)
    • group_psd_{control,epilepsy}.npy       shape (22, F)
    • group_psd_freqs.npy                    shape (F,)
    • group_psd_{control,epilepsy}.png       PSD curves (selected chans)
    • topomap_{band}_{control,epilepsy}.png  band-power scalp maps
"""

from pathlib import Path
import numpy as np
import pickle, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne
import argparse
import time

CORE_CHS = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
             "T1","T3","C3","Cz","C4","T4","T2",
             "T5","P3","Pz","P4","T6","O1","Oz","O2"]

FS_TARGET = 250.0  # resample_hz used in preprocessing; change if you used a different value

def plot_group_time_curves(GA_t, chs, fs, name, out_png, sel=("Fz","Cz","Pz","O1","O2")):
    """
    GA_t: (C, T) group grand-average in time domain (NaN for padded chans)
    chs: list of channel names in CORE_CHS order
    fs:  sampling rate (Hz)
    name: group name for title
    out_png: path to save figure
    """
    # Removed unnecessary local imports of numpy and matplotlib

    t = np.arange(GA_t.shape[1]) / fs
    present = [c for c in sel if c in chs]
    idx = [chs.index(c) for c in present]

    fig = plt.figure(figsize=(10, 5))
    for i, c in zip(idx, present):
        y = GA_t[i]
        if np.isfinite(y).any():
            plt.plot(t, y, label=c, alpha=0.9)
    plt.xlabel("Time (s)")
    # --- CORRECTED Y-AXIS LABEL ---
    plt.ylabel(r"Amplitude ($\mu V$)")
    plt.title(f"Group grand-average (time domain) — {name}")
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def infer_label_from_path(p: Path) -> int:
    s = str(p).replace("\\","/").lower()
    return 1 if "/00_epilepsy/" in s else 0  # 1=epilepsy, 0=control

def load_subject(prefix: Path):
    """
    prefix: path without the suffix, e.g. /.../aaaaaanr_s001_t001   (no _epochs.npy)
    returns: X (E,C,T), mask (C,), chs (list[str]), fs (float)
    """
    X = np.load(prefix.parent / f"{prefix.name}_epochs.npy")  # (E,C,T)
    mask_path = prefix.parent / f"{prefix.name}_present_mask.npy"
    mask = np.load(mask_path).astype(bool) if mask_path.exists() else np.ones(X.shape[1], bool)

    info_pkl = prefix.parent / f"{prefix.name}_info.pkl"
    with open(info_pkl, "rb") as f:
        info = pickle.load(f)
    chs = list(info["ch_names"])
    fs = float(info["sfreq"])
    return X, mask, chs, fs

def subject_time_mean(X, mask):
    X = X.astype(np.float64, copy=False)
    X[:, ~mask, :] = np.nan
    return np.nanmean(X, axis=0)  # (C,T)

def subject_psd_mean(X, mask, fs, nperseg=512, noverlap=256):
    E, C, T = X.shape
    freqs = None
    psd = np.full((C, 0), np.nan)
    rows = []
    for c in range(C):
        if not mask[c]:
            rows.append(None)
            continue
        pxx_list = []
        for e in range(E):
            f, Pxx = welch(X[e, c, :], fs=fs, nperseg=min(nperseg, T), noverlap=min(noverlap, T//2))
            pxx_list.append(Pxx)
        pxx_arr = np.vstack(pxx_list)  # (E,F)
        if freqs is None:
            freqs = f
        rows.append(pxx_arr.mean(axis=0))
    C_out, F = C, len(freqs)
    psd = np.full((C_out, F), np.nan)
    for c, row in enumerate(rows):
        if row is not None:
            psd[c] = row
    return psd, freqs

def topomap_band(power_c, chs, band_name, out_png, vlim="auto"):
    """
    power_c: array (C,) band power per channel (can include NaN -> ignored)
    chs: channel names in CORE_CHS order
    Adds approximate positions for T1/T2 (FT9/FT10-like).
    """
    std = mne.channels.make_standard_montage("standard_1020")
    pos = std.get_positions()["ch_pos"].copy()
    # Add T1/T2 approx
    if "T1" in chs and "T1" not in pos:
        pos["T1"] = np.array([-0.0840759, 0.0145673, -0.050429])
    if "T2" in chs and "T2" not in pos:
        pos["T2"] = np.array([ 0.0841131, 0.0143647, -0.050538])

    chs_with_pos = [c for c in chs if c in pos]
    idx = [chs.index(c) for c in chs_with_pos]
    data_keep = power_c[idx]

    dig = mne.channels.make_dig_montage(
        ch_pos={c: pos[c] for c in chs_with_pos}, coord_frame="head"
    )
    info = mne.create_info(chs_with_pos, sfreq=250.0, ch_types="eeg")
    info.set_montage(dig, match_case=False)

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    im, _ = mne.viz.plot_topomap(
        data_keep, info, axes=ax, show=False, names=chs_with_pos,
        contours=0, cmap="RdBu_r", sphere=0.09
    )
    ax.set_title(f"{band_name} band power")
    if vlim != "auto":
        im.set_clim(*vlim)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07)
    cbar.ax.set_xlabel(r"Power ($\mu V^2/Hz$)") # Added units to colorbar
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main(data_pp_dir: str, out_dir: str, max_subjects: int = None,
             plot_channels=("Fz","Cz","Pz","O1","O2"),
             bands=(("theta",(4,7)), ("alpha",(8,12)), ("beta",(13,30)))):
    root = Path(data_pp_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    global_start_time = time.time() # <-- ADDED: Start global timer
    # Find all subject prefixes
    epochs = sorted(root.rglob("*_epochs.npy"))
    prefixes = [Path(str(p).rsplit("_epochs.npy",1)[0]) for p in epochs]

    groups = {0: [], 1: []}   # 0=control, 1=epilepsy
    freqs_ref = None
    chs_ref = CORE_CHS  # postproc should align to CORE_CHS

    # Per-subject aggregation
    count = 0
    for pref in prefixes:
        try:
            start_time = time.time()
            print(f"Processing subject: {pref.name}")

            X, mask, chs, fs = load_subject(pref)
            if len(chs) != len(CORE_CHS):
                # Safety: reorder if needed
                order = [chs.index(c) for c in CORE_CHS if c in chs]
                X = X[:, order, :]
                mask = mask[order]
                chs = [chs[i] for i in order]
            subj_time = subject_time_mean(X, mask)              # (C,T)
            subj_psd, freqs = subject_psd_mean(X, mask, fs)      # (C,F)

            if freqs_ref is None:
                freqs_ref = freqs
            elif len(freqs) != len(freqs_ref) or not np.allclose(freqs, freqs_ref):
                print(f"[WARN] {pref.name}: PSD freqs differ -> skipping subject.")
                continue

            label = infer_label_from_path(pref)
            groups[label].append((subj_time, subj_psd))


            elapsed = time.time() - start_time
            print(f"Finished {pref.name} in {elapsed:.1f} seconds")
            
            count += 1
            if max_subjects and count >= max_subjects:
                break
        except Exception as e:
            print(f"[ERR] {pref}: {e}")

    if not groups[0] and not groups[1]:
        raise RuntimeError("No subjects aggregated. Check data_pp_dir.")

    # Helper: group mean with NaN-safe averaging
    def nanmean_stack(items, axis=0, out_shape=None):
    #"""NaN-safe mean over a list. If empty, return NaNs of out_shape."""
        if len(items) == 0:
            if out_shape is None:
                raise ValueError("nanmean_stack called with empty items and no out_shape.")
            return np.full(out_shape, np.nan)
        arr = np.stack(items, axis=0)
        return np.nanmean(arr, axis=axis)

    # --- compute & save per-group grand averages ---
    saved = {}
    for label, name in [(0,"control"), (1,"epilepsy")]:
        n_subj = len(groups[label])
        print(f"[INFO] {name}: {n_subj} subjects aggregated")
        if n_subj == 0:
            # make placeholder NaN arrays so later code can still run
            GA_t = np.full((len(CORE_CHS), 1), np.nan)
            GA_p = np.full((len(CORE_CHS), len(freqs_ref) if freqs_ref is not None else 1), np.nan)
            saved[name] = dict(time=GA_t, psd=GA_p)
            continue

        times = [t for (t, p) in groups[label]]
        psds  = [p for (t, p) in groups[label]]
        # Provide shapes so empty lists won’t warn (we guarded above anyway)
        GA_t = nanmean_stack(times, axis=0, out_shape=times[0].shape)     # (C,T)
        GA_p = nanmean_stack(psds,  axis=0, out_shape=psds[0].shape)      # (C,F)

        np.save(out / f"group_time_{name}.npy", GA_t)
        np.save(out / f"group_psd_{name}.npy",  GA_p)
        # Save time-domain figure
        plot_group_time_curves(
            GA_t, CORE_CHS, FS_TARGET, name,
            out / f"group_time_{name}.png"
        )

        if freqs_ref is not None:
            np.save(out / f"group_psd_freqs.npy",   freqs_ref)
        saved[name] = dict(time=GA_t, psd=GA_p)

        # PSD curves (selected channels)
        ch_to_idx = {c:i for i,c in enumerate(CORE_CHS)}
        fig = plt.figure(figsize=(10,5))
        for ch in plot_channels:
            if ch in ch_to_idx:
                P = GA_p[ch_to_idx[ch]]
                if np.isfinite(P).any():
                    plt.semilogy(freqs_ref, P, label=ch)
        plt.xlabel("Frequency (Hz)")
        # --- CORRECTED Y-AXIS LABEL ---
        plt.ylabel(r"Power Spectral Density ($\mu V^2/Hz$)")
        plt.title(f"Group grand-average PSD ({name})")
        plt.xlim(0, 100)
        plt.legend()
        plt.tight_layout()
        fig.savefig(out / f"group_psd_{name}.png", dpi=150)
        plt.close(fig)

        # Topomaps per band
        for bname, (fmin,fmax) in bands:
            if freqs_ref is None:
                continue
            idx = np.where((freqs_ref >= fmin) & (freqs_ref <= fmax))[0]
            if idx.size == 0:
                print(f"[WARN] {name}: band {bname} ({fmin}-{fmax}) not in frequency grid; skipping topomap.")
                continue
            band_power = np.nanmean(GA_p[:, idx], axis=1)  # (C,)
            topomap_band(band_power, CORE_CHS, f"{bname} ({fmin}-{fmax} Hz)",
                         out / f"topomap_{bname}_{name}.png")


    print(f"Saved group outputs in: {out}")

    # --- NEW: DIFF (Epilepsy - Control) ---
    if "epilepsy" in saved and "control" in saved:
        diff_psd = saved["epilepsy"]["psd"] - saved["control"]["psd"]  # (C,F)
        np.save(out / "group_psd_diff_epilepsy_minus_control.npy", diff_psd)

        # PSD difference curves on selected channels (linear y, can also do semilogy)
        ch_to_idx = {c:i for i,c in enumerate(CORE_CHS)}
        fig = plt.figure(figsize=(10,5))
        for ch in plot_channels:
            if ch in ch_to_idx:
                D = diff_psd[ch_to_idx[ch]]
                if np.isfinite(D).any():
                    plt.plot(freqs_ref, D, label=ch)  # signed difference
        plt.axhline(0, color="k", lw=1, ls="--")
        plt.xlabel("Frequency (Hz)")
        # --- CORRECTED Y-AXIS LABEL ---
        plt.ylabel(r"PSD difference ($\mu V^2/Hz$) (Epilepsy - Control)")
        plt.title("Group PSD difference (Epilepsy - Control)")
        plt.xlim(0, 100)
        plt.legend()
        plt.tight_layout()
        fig.savefig(out / "group_psd_diff_curves.png", dpi=150)
        plt.close(fig)

        # Topomap differences per band with symmetric color limits around 0
        for bname, (fmin,fmax) in bands:
            idx = np.where((freqs_ref >= fmin) & (freqs_ref <= fmax))[0]
            diff_band = np.nanmean(diff_psd[:, idx], axis=1)  # (C,)

            # symmetric vlim for comparable color scaling
            vmax = np.nanmax(np.abs(diff_band))
            vlim = (-vmax, vmax) if np.isfinite(vmax) and vmax > 0 else "auto"

            # reuse topomap helper; pass vlim via wrapper
            def topomap_band_with_vlim(power_c, chs, band_name, out_png, vlim):
                std = mne.channels.make_standard_montage("standard_1020")
                pos = std.get_positions()["ch_pos"].copy()
                if "T1" in chs and "T1" not in pos:
                    pos["T1"] = np.array([-0.0840759, 0.0145673, -0.050429])
                if "T2" in chs and "T2" not in pos:
                    pos["T2"] = np.array([ 0.0841131, 0.0143647, -0.050538])
                chs_with_pos = [c for c in chs if c in pos]
                idx_keep = [chs.index(c) for c in chs_with_pos]
                data_keep = power_c[idx_keep]
                dig = mne.channels.make_dig_montage(
                    ch_pos={c: pos[c] for c in chs_with_pos}, coord_frame="head"
                )
                info = mne.create_info(chs_with_pos, sfreq=250.0, ch_types="eeg")
                info.set_montage(dig, match_case=False)
                fig, ax = plt.subplots(figsize=(4.8, 5.2))
                im, _ = mne.viz.plot_topomap(
                    data_keep, info, axes=ax, show=False, names=chs_with_pos,
                    contours=0, cmap="RdBu_r", sphere=0.09
                )
                ax.set_title(f"{band_name} (Epi - Ctrl)")
                if vlim != "auto":
                    im.set_clim(*vlim)
                cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07)
                cbar.ax.set_xlabel("Δ Power")
                Path(out_png).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out / f"topomap_{bname}_diff_epi_minus_ctrl.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

            topomap_band_with_vlim(diff_band, CORE_CHS, f"{bname} ({fmin}-{fmax} Hz)", 
                                     out / f"topomap_{bname}_diff_epi_minus_ctrl.png", vlim=vlim)


    global_end_time = time.time() # <-- ADDED: End global timer
    total_elapsed = global_end_time - global_start_time
    print(f"\nTotal processing time for all files: {total_elapsed:.1f} seconds") # <-- ADDED: Total time output
    print(f"All outputs saved in: {out}") # <-- MODIFIED: Final confirmation

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pp_dir", required=True, help="Folder with processed outputs (data_pp)")
    ap.add_argument("--out_dir", required=True, help="Where to save group averages/plots") 
    ap.add_argument("--max_subjects", type=int, default=None, help="Limit total subjects")
    args = ap.parse_args()
    main(args.data_pp_dir, args.out_dir, max_subjects=args.max_subjects)