from pathlib import Path
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import welch

CANON = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
         "T1","T3","C3","Cz","C4","T4","T2",
         "T5","P3","Pz","P4","T6","O1","Oz","O2"]

def _load_info(pkl):
    with open(pkl, "rb") as f:
        info = pickle.load(f)
    return list(info["ch_names"]), float(info["sfreq"])

def _align_to_canon(X, chs, canon=CANON):
    E, C, T = X.shape
    Y = np.full((E, len(canon), T), np.nan, dtype=X.dtype)
    idx = {c: i for i, c in enumerate(chs)}
    for j, c in enumerate(canon):
        if c in idx:
            Y[:, j, :] = X[:, idx[c], :]
    return Y

def _per_patient_mean_psd(epochs_npy, info_pkl, nperseg=256):
    X = np.load(epochs_npy)
    chs, fs = _load_info(info_pkl)
    Xc = _align_to_canon(X, chs)
    # time-domain grand average
    ga = np.nanmean(Xc, axis=0)
    # frequency-domain per epoch
    E, C, T = Xc.shape
    psds = []
    for epoch in Xc:
        psd_epoch = []
        for ch_signal in epoch:
            if not np.isnan(ch_signal).all():
                freqs, Pxx = welch(ch_signal, fs=fs, nperseg=nperseg)
            else:
                freqs, Pxx = welch(np.zeros(T), fs=fs, nperseg=nperseg)
                Pxx[:] = np.nan
            psd_epoch.append(Pxx)
        psds.append(psd_epoch)
    psds = np.array(psds)  # (E, C, F)
    mean_psd = np.nanmean(psds, axis=0)  # (C, F)
    return ga, mean_psd, fs, freqs

def _class_from_labels(labels_npy):
    y = np.load(labels_npy)
    return int(np.round(y.mean()))

def _plot_time(ga, fs, title, out_png):
    sel = [c for c in ["Fz", "Cz", "Pz", "O1", "O2"] if c in CANON]
    idx = [CANON.index(c) for c in sel]
    t = np.arange(ga.shape[1]) / fs
    plt.figure(figsize=(10, 5))
    for i, c in zip(idx, sel):
        plt.plot(t, ga[i], label=c)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z-score)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def _plot_psd(mean_psd, freqs, title, out_png):
    sel = [c for c in ["Fz","Cz","Pz","O1","O2"] if c in CANON]
    idx = [CANON.index(c) for c in sel]
    plt.figure(figsize=(10,5))
    for i, c in zip(idx, sel):
        plt.semilogy(freqs, mean_psd[i], label=c)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (a.u.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

def run(root_pp, out_dir, verbose=False):
    root = Path(root_pp)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_class_time = defaultdict(list)
    by_class_psd = defaultdict(list)
    fs_set = set()
    freq_set = set()
    skipped = 0

    for info_pkl in root.rglob("*_info.pkl"):
        stem = info_pkl.stem.replace("_info", "")
        epochs = info_pkl.with_name(f"{stem}_epochs.npy")
        labels = info_pkl.with_name(f"{stem}_labels.npy")
        if not (epochs.exists() and labels.exists()):
            skipped += 1
            if verbose:
                print(f"Skipping incomplete dataset for {stem}")
            continue
        try:
            ga, mean_psd, fs, freqs = _per_patient_mean_psd(epochs, info_pkl)
            c = _class_from_labels(labels)
            by_class_time[c].append(ga)
            by_class_psd[c].append(mean_psd)
            fs_set.add(fs)
            freq_set.add(len(freqs))
            if verbose:
                print(f"Processed subject {stem}, class {c}")
        except Exception as e:
            skipped += 1
            print(f"Error processing {info_pkl}: {e}")

    if skipped > 0 and verbose:
        print(f"Skipped {skipped} files due to errors or incompleteness")
    if len(fs_set) != 1:
        raise ValueError(f"Mixed sampling rates found: {fs_set}")
    if len(freq_set) != 1:
        raise ValueError(f"Mixed PSD bins found: {freq_set}")
    fs = fs_set.pop()

    # Time-domain grand averages
    grp_time = {c: np.nanmean(np.stack(v, 0), 0) for c, v in by_class_time.items()}
    all_ga_time = np.nanmean(np.stack([g for L in by_class_time.values() for g in L], 0), 0)

    # Frequency-domain (PSD) grand averages
    grp_psd = {c: np.nanmean(np.stack(v, 0), 0) for c, v in by_class_psd.items()}
    all_ga_psd = np.nanmean(np.stack([g for L in by_class_psd.values() for g in L], 0), 0)
    freqs = freqs

    # Save time-domain arrays + plots
    if 0 in grp_time:
        np.save(out / "grand_avg_controls.npy", grp_time[0])
        _plot_time(grp_time[0], fs, "Grand average (controls)", out / "grand_avg_controls.png")
    if 1 in grp_time:
        np.save(out / "grand_avg_epilepsy.npy", grp_time[1])
        _plot_time(grp_time[1], fs, "Grand average (epilepsy)", out / "grand_avg_epilepsy.png")
    np.save(out / "grand_avg_all.npy", all_ga_time)
    _plot_time(all_ga_time, fs, "Grand average (all patients)", out / "grand_avg_all.png")

    # Save frequency-domain arrays + plots
    if 0 in grp_psd:
        np.save(out / "grand_psd_controls.npy", grp_psd[0])
        _plot_psd(grp_psd[0], freqs, "Grand PSD (controls)", out / "grand_psd_controls.png")
    if 1 in grp_psd:
        np.save(out / "grand_psd_epilepsy.npy", grp_psd[1])
        _plot_psd(grp_psd[1], freqs, "Grand PSD (epilepsy)", out / "grand_psd_epilepsy.png")
    np.save(out / "grand_psd_all.npy", all_ga_psd)
    _plot_psd(all_ga_psd, freqs, "Grand PSD (all patients)", out / "grand_psd_all.png")

    print(f"Grand averages saved to: {out}")
    return grp_time, all_ga_time, grp_psd, all_ga_psd, fs, freqs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute grand average EEG and PSD for epilepsy and control groups")
    parser.add_argument("--root", required=True, help="Preprocessed data root directory")
    parser.add_argument("--out", required=True, help="Output directory for grand averages and plots")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    run(args.root, args.out, verbose=args.verbose)
