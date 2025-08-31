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
    A["Raw EEG (EDF) data_raw/DATA/..."]
    B["Standardized Raw (EEG selected, names cleaned, core 10-20)"]
    C["Preprocessed Continuous (montage, CAR, notch, band-pass, ICA)"]
    D["Harmonized Continuous (resample 250 Hz, crop 10s if control)"]
    E["Epochs (2s windows)"]
    F["Clean Epochs (z-score, artifact rejection)"]
    G["Labels (0/1 per epoch, from folder)"]
    H["Saved Arrays: data_pp/*_epochs.npy, *_labels.npy, *_raw.npy, *_info.pkl"]
    P1["PSD Before (figures/psd/*_PSD_before.png)"]
    P2["PSD After (figures/psd/*_PSD_after.png)"]
    I["Downstream: features + connectivity → graphs, models"]
    J["Notebooks (exploration, figures)"]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    A -.-> G
    F --> H
    B -.-> P1
    C -.-> P2
    H --> I
    I --> J
