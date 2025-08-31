# Thesis: Graph-Based Self-Supervised Learning for Epilepsy Detection

## Repo structure
- `data_raw/`   → raw TUEP EDFs (ignored by git)
- `data_pp/`   → preprocessed numpy/pkl outputs (ignored by git)
- `figures/`   → plots (ignored by git)
- `src/`       → Python scripts
- `notebooks/` → Jupyter notebooks for exploratory analysis

---

## Preprocessing Workflow

```mermaid
flowchart LR
    A[Raw EEG (EDF) data_raw/DATA/...] -->|select EEG, clean names, pick core 10-20| B[Standardized Raw]
    B -->|set montage, CAR, notch, band-pass, (optional ICA)| C[Preprocessed Continuous]
    C -->|resample to 250 Hz (optional crop 10s for controls)| D[Harmonized Continuous]
    D -->|fixed 2s windows| E[Epochs]
    E -->|artifact rejection (peak-to-peak, 95th pct)| F[Clean Epochs (z-score)]
    A -.->|group label (folder name)| G[Labels (0/1) per epoch]
    F --> H[Save ML-ready arrays: data_pp/*_epochs.npy, *_labels.npy, *_raw.npy, *_info.pkl]
    B -.->|PSD BEFORE (QC)| P1[figures/psd/*_PSD_before.png]
    C -.->|PSD AFTER (QC)| P2[figures/psd/*_PSD_after.png]
    H --> I[Downstream: features, connectivity → graphs, models]
    I --> J[Notebooks for figures (notebooks/*.ipynb)]




Usage:
```bash
python src/preprocess_single.py \
  --edf "data_raw/DATA/01_no_epilepsy/aaaaafiy/s002_2010/01_tcp_ar/aaaaafiy_s002_t001.edf" \
  --out "data_pp" \
  --psd_dir "figures/psd"

