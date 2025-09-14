# Graph-Based Self-Supervised Learning for Epilepsy Detection

This repository contains the code, preprocessing pipeline, and experiments for my MSc thesis.

## Dataset
- TUH EEG Epilepsy Corpus (TUEP v2.0.1)  
- 100 epilepsy subjects, 100 non-epilepsy subjects.

## Preprocessing Pipeline
1. Load EDF, keep EEG only
2. Clean channel names
3. Core 10–20 channel selection
4. Apply montage (+T1/T2)
5. Common average reference
6. Notch @ 60 Hz
7. Bandpass (0.5–100 Hz)
8. ICA for artifact removal
9. PSD plots
10. Resample (250 Hz)
11. Crop first 10s (non-epilepsy only)
12. Epoch into 2s windows
13. Artifact rejection (95th percentile)
14. Z-score normalization
15. Save outputs

## Files
- `preprocess_single_edf.ipynb` → interactive preprocessing with plots
- `preprocess_single_edf.py` → script for batch preprocessing
- Output: `_raw.npy`, `_epochs.npy`, `_labels.npy`, `_info.pkl`

## Citation
If you use this code, please cite:  
Zhao et al. (2023). *Self-supervised learning for epileptic EEG analysis using graph neural networks.* arXiv:2311.03764
