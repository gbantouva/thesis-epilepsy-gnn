# src/psd_batch.py
from pathlib import Path
import argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from collections import defaultdict

CANON = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T1","T3","C3","Cz","C4","T4","T2","T5","P3","Pz","P4","T6","O1","Oz","O2"]
PLOT_CHS = ["Fz","Cz","Pz","O1","O2"]

def _load(epochs_npy, info_pkl):
    X = np.load(epochs_npy)  # (E,C,T)
    with open(info_pkl, "rb") as f: info = pickle.load(f)
    return X, float(info["sfreq"]), list(info["ch_names"])

def _relative_psd(f, pxx, lo=0.5, hi=45.0):
    mask = (f >= lo) & (f <= hi)
    area = np.trapz(pxx[..., mask], f[mask], axis=-1)[..., None]
    area = np.where(area == 0, 1.0, area)
    return pxx / area

def _align_to_canon(arr, chs, axis_ch=0):
    """arr shape (C, F) or (N,C,F) → align channel axis to CANON, pad with NaN."""
    ch2i = {c:i for i,c in enumerate(chs)}
    shape = list(arr.shape)
    shape[axis_ch] = len(CANON)
    out = np.full(shape, np.nan, dtype=arr.dtype)
    if arr.ndim == 2:
        for j,c in enumerate(CANON):
            if c in ch2i: out[j] = arr[ch2i[c]]
    else:
        for j,c in enumerate(CANON):
            if c in ch2i: out[:, j] = arr[:, ch2i[c]]
    return out

def _psd_per_patient(epochs_npy, info_pkl, nperseg=256, relative=True):
    X, fs, chs = _load(epochs_npy, info_pkl)   # (E,C,T)
    E, C, T = X.shape
    psds = []
    for e in range(E):
        f, pxx = welch(X[e], fs=fs, nperseg=min(nperseg, T), axis=-1, average="median")  # (C,F)
        psds.append(pxx)
    psd_ch = np.mean(np.stack(psds, axis=0), axis=0)    # (C,F)
    if relative:
        psd_ch = _relative_psd(f, psd_ch)
    psd_ch = _align_to_canon(psd_ch, chs)               # → (22,F)
    return f, psd_ch, fs

def _class_from_labels(labels_npy):
    y = np.load(labels_npy)              # per-epoch labels (constant per subject)
    return int(np.round(y.mean()))       # 0 control / 1 epilepsy

def run(root_pp, out_dir, nperseg=256, relative=True):
    root = Path(root_pp); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fs_set = set()
    by_class = defaultdict(list)   # class → list of (22,F)
    all_patients = []

    for info_pkl in root.rglob("*_info.pkl"):
        stem = info_pkl.stem.replace("_info","")
        epochs = info_pkl.with_name(f"{stem}_epochs.npy")
        labels = info_pkl.with_name(f"{stem}_labels.npy")
        if not (epochs.exists() and labels.exists()): continue
        try:
            f, psd_ch, fs = _psd_per_patient(epochs, info_pkl, nperseg=nperseg, relative=relative)
            fs_set.add(fs)
            cls = _class_from_labels(labels)
            by_class[cls].append(psd_ch)   # (22,F)
            all_patients.append(psd_ch)
        except Exception as e:
            print("skip", info_pkl, e)

    assert len(fs_set) == 1, f"Mixed sampling rates found: {fs_set}"
    # group means
    grp = {c: np.nanmean(np.stack(v, axis=0), axis=0) for c, v in by_class.items()}  # (22,F)
    overall = np.nanmean(np.stack(all_patients, axis=0), axis=0)                     # (22,F)

    # save arrays
    np.savez_compressed(out/"psd_groups.npz", f=f, controls=grp.get(0), epilepsy=grp.get(1),
                        overall=overall, channels=np.array(CANON))

    # plots: global mean across channels + selected channels overlay
    def _plot_global(psd, title, fname):
        if psd is None: return
        g = np.nanmean(psd, axis=0)   # (F,)
        plt.figure(figsize=(8,4)); plt.plot(f, g)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (relative)" if relative else "PSD (a.u.)")
        plt.title(title); plt.tight_layout()
        plt.savefig(out/fname, dpi=150); plt.close()

    def _plot_selected(psd, title, fname):
        if psd is None: return
        plt.figure(figsize=(9,5))
        for ch in PLOT_CHS:
            i = CANON.index(ch)
            if np.all(~np.isfinite(psd[i])): continue
            plt.plot(f, psd[i], label=ch)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (relative)" if relative else "PSD (a.u.)")
        plt.title(title); plt.legend(); plt.tight_layout()
        plt.savefig(out/fname, dpi=150); plt.close()

    _plot_global(grp.get(0), "Group PSD (controls, global)", "controls_global.png")
    _plot_global(grp.get(1), "Group PSD (epilepsy, global)", "epilepsy_global.png")
    _plot_global(overall,     "Group PSD (overall, global)", "overall_global.png")

    _plot_selected(grp.get(0), "Group PSD (controls, selected chs)", "controls_selected.png")
    _plot_selected(grp.get(1), "Group PSD (epilepsy, selected chs)", "epilepsy_selected.png")

    print(f"[batch] Saved to {out} | patients={len(all_patients)} | fs={list(fs_set)[0]} Hz")
    return grp, overall, f

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grand-average PSD across patients")
    ap.add_argument("--root", required=True, help="preprocessed root (e.g., data_pp)")
    ap.add_argument("--out",  required=True, help="output folder for arrays/figures")
    ap.add_argument("--nperseg", type=int, default=256)
    ap.add_argument("--relative", action="store_true"); ap.add_argument("--absolute", dest="relative", action="store_false")
    args = ap.parse_args()
    run(args.root, args.out, args.nperseg, args.relative)
