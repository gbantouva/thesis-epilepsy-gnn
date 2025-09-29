# Graph-based Self-Supervised Learning for Epilepsy Detection

## Overview
This repository contains the source code and documentation for the thesis: **Graph-based Self-Supervised Learning for Epilepsy Detection**, utilizing the **TUH EEG Epilepsy Corpus (TUEP)**.

The project begins with a robust and scalable pipeline to preprocess raw EEG data into clean, normalized epochs, which is the necessary input for subsequent connectivity analysis and Graph Neural Network (GNN) modeling.

## 📁 Repository Structure

| Folder | Purpose | Status |
| :--- | :--- | :--- |
| `data_raw/` | **Raw Input:** TUH EDF files (Ignored by Git). | Input |
| `data_pp/` | **Preprocessed Output:** Clean epochs (`.npy`), labels, and metadata (`.pkl`). | Output |
| `figures/psd/` | Quality Control (QC) plots of Power Spectral Density (PSD) figures. | Output |
| `src/` | **Source Code:** Core Python scripts for all thesis steps. | Core |
| `notebooks/` | Jupyter notebooks for Exploratory Data Analysis (EDA). | In Progress |

***

## ⚙️ Preprocessing Pipeline (Step 1)

The pipeline is split into two files for maximum **modularity and reusability**.

### 1. `src/preprocess_core.py` (The Engine)
This file contains the **core scientific logic** and hyperparameters for EEG processing.

* **Standardization:** Channel selection (`CORE_CHS`) and 10-20 montage assignment (including T1/T2 approximation).
* **Denoising:** Average Reference, $\text{Notch}$ ($\text{60 Hz}$), and $\text{Bandpass}$ ($\text{0.5-100 Hz}$) filtering.
* **Transformation:** Resampling to $\text{250 Hz}$ and cropping the first $\text{10 s}$ of non-epileptic (control) files.
* **Artifact Removal:** $\text{2.0 s}$ fixed-length epoching followed by amplitude-based rejection ($\text{95th percentile}$).
* **Normalization:** **Per-epoch, per-channel Z-score** to stabilize features for the GNN.

### 2. `src/preprocess_single.py` (The Interface)
This is a **Command-Line Interface (CLI)** script used for testing and quality assurance of the core pipeline. It handles file I/O and saves QC plots.

**Output Files (saved per subject in `data_pp/`):**
* `[pid]_epochs.npy`
* `[pid]_labels.npy`
* `[pid]_info.pkl` (MNE metadata)

***

## 🚀 How to Run and Test

The following instructions demonstrate how to run the preprocessing on a single file using the CLI script.

### Prerequisites

You must have the raw EDF data downloaded and organized under `data_raw/` and the necessary Python packages installed (`mne`, `numpy`, `matplotlib`, etc.).

### Single-File Quality Control Test

Run the `preprocess_single.py` script from your repository root using the command below. **Note:** The example uses Windows PowerShell syntax (`\`` for line continuation) and a placeholder path—adjust as necessary for your OS and file location.

```powershell
# Run from the repository root (e.g., C:\...\thesis-epilepsy-gnn)
python src\preprocess_single.py `
    --edf "data_raw\DATA\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001.edf" `
    --out "C:\Users\georg\Documents\GitHub\thesis-epilepsy-gnn\data_pp" `
    --psd_dir "C:\Users\georg\Documents\GitHub\thesis-epilepsy-gnn\figures\psd"