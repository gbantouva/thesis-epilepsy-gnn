# Step 2: Connectivity Analysis - Complete Guide

## Table of Contents
1. [What is This?](#what-is-this)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Understanding Your Results](#understanding-your-results)
5. [Troubleshooting](#troubleshooting)
6. [Next Steps](#next-steps)

---

## What is This?

### The Goal
You want to find **how brain regions communicate** in epilepsy patients vs controls.

### The Method: MVAR → DTF/PDC

Think of your brain like a network of 22 radio stations (your EEG channels):
- Some stations **broadcast** signals
- Other stations **receive** and are influenced by those signals
- **MVAR (Multivariate AutoRegressive)** models find these broadcasting patterns
- **DTF and PDC** measure the strength and direction of influence

### DTF vs PDC (Simple Explanation)

| Measure | What it finds | Example |
|---------|---------------|---------|
| **DTF** | Total influence (including indirect paths) | "How much does region A affect region B overall?" (maybe through C, D, etc.) |
| **PDC** | Direct influence only | "Is there a direct wire from A to B?" |

**For your thesis**: Both are useful! Epilepsy often shows abnormal connectivity patterns where certain regions start "driving" the whole network.

---

## Installation

### Step 1: Check Your Python Environment

Open VSCode terminal and run:
```bash
python --version
```

You should see: `Python 3.12.7` or similar.

### Step 2: Install Required Packages

**Option A: Try this first**
```bash
pip install numpy scipy matplotlib tqdm seaborn
```

**Option B: If you get errors with Python 3.12** (dyconnmap might not support it yet)

Create a new environment with Python 3.10:
```bash
# If using conda:
conda create -n thesis python=3.10
conda activate thesis

# If using venv:
python -m venv thesis_env
thesis_env\Scripts\activate  # Windows
source thesis_env/bin/activate  # Linux/Mac

# Install packages
pip install numpy scipy matplotlib tqdm seaborn mne
```

### Step 3: Verify Installation

Create a file called `test_install.py`:
```python
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

print("✓ All packages installed successfully!")
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {scipy.__version__}")
```

Run it: `python test_install.py`

---

## Quick Start

### Single File Test (Recommended First Step)

1. **Find one preprocessed file** from your data. For example:
   ```
   F:\October-Thesis\thesis-epilepsy-gnn\test\data_pp\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_epochs.npy
   ```

2. **Run connectivity analysis**:
   ```bash
   python connectivity_analysis.py \
     --epoch_file "F:\October-Thesis\thesis-epilepsy-gnn\test\data_pp\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_epochs.npy" \
     --output_dir "F:\October-Thesis\thesis-epilepsy-gnn\connectivity_results\test"
   ```

3. **What to expect**:
   - Processing time: ~30 seconds to 2 minutes per patient
   - Output: Console shows progress, MVAR order selection, epoch processing

4. **Check your results** (see [Understanding Your Results](#understanding-your-results))

### Batch Processing (All Patients)

Once the single file works, process all patients:

```bash
python connectivity_batch.py \
  --data_dir "F:\October-Thesis\thesis-epilepsy-gnn\test\data_pp" \
  --output_dir "F:\October-Thesis\thesis-epilepsy-gnn\connectivity_results" \
  --max_files 10
```

**Parameters**:
- `--data_dir`: Your preprocessed data folder (where *_epochs.npy files are)
- `--output_dir`: Where to save connectivity results
- `--max_files 10`: Process only 10 files (for testing). Remove this to process all.
- `--skip_existing`: Add this to skip already-processed files (useful if script crashes)

**Estimated time**: ~1 minute per patient × number of patients

---

## Understanding Your Results

### Output Files Structure

For each patient, you get:

```
connectivity_results/
└── 00_epilepsy/
    └── aaaaaanr/
        └── s001_2003/
            └── 02_tcp_le/
                ├── aaaaaanr_s001_t001_dtf_delta.npy          # DTF connectivity in delta band
                ├── aaaaaanr_s001_t001_dtf_theta.npy          # DTF in theta band
                ├── aaaaaanr_s001_t001_dtf_alpha.npy          # DTF in alpha band
                ├── aaaaaanr_s001_t001_dtf_beta.npy           # DTF in beta band
                ├── aaaaaanr_s001_t001_dtf_gamma.npy          # DTF in gamma band
                ├── aaaaaanr_s001_t001_pdc_delta.npy          # PDC connectivity in delta band
                ├── aaaaaanr_s001_t001_pdc_theta.npy          # PDC in theta band
                ├── aaaaaanr_s001_t001_pdc_alpha.npy          # PDC in alpha band
                ├── aaaaaanr_s001_t001_pdc_beta.npy           # PDC in beta band
                ├── aaaaaanr_s001_t001_pdc_gamma.npy          # PDC in gamma band
                ├── aaaaaanr_s001_t001_connectivity_metadata.json  # Info about processing
                └── aaaaaanr_s001_t001_connectivity_visualization.png  # Visual check
```

### Understanding the .npy Files

Each `*_dtf_*.npy` or `*_pdc_*.npy` file contains:
- **Shape**: `(n_epochs, 22, 22)`
- **Meaning**: For each 2-second epoch, a 22×22 matrix where:
  - **Row i, Column j** = influence from channel j → channel i
  - Values range from 0 to 1 (higher = stronger connection)

**Example**: If `dtf_alpha.npy[10, 5, 8] = 0.75`, it means:
- In epoch 10 (the 11th 2-second window)
- In the alpha frequency band (8-13 Hz)
- Channel 8 has a strong (0.75) influence on Channel 5

### Checking the Visualization

Open the `*_connectivity_visualization.png` file:
- **Top row**: DTF matrices (total influence)
- **Bottom row**: PDC matrices (direct influence)
- **Columns**: One per frequency band (delta, theta, alpha, beta, gamma)

**What to look for**:
- ✓ **Hot colors (red/yellow)**: Strong connections
- ✓ **Patterns**: You should see some structure (not all zeros, not all random)
- ✓ **Diagonal**: Should NOT be too bright (channels shouldn't strongly "influence themselves")

### Checking the Metadata

Open the `*_connectivity_metadata.json` file:
```json
{
  "patient_id": "aaaaaanr_s001_t001",
  "n_epochs": 150,
  "n_channels": 22,
  "channel_names": ["Fp1", "Fp2", "F7", ...],
  "sfreq": 250.0,
  "avg_model_order": 12.5,
  "std_model_order": 2.1,
  "frequency_bands": {
    "delta": [0.5, 4],
    "theta": [4, 8],
    ...
  }
}
```

**What to check**:
- `n_epochs`: Should match your preprocessed data
- `avg_model_order`: Typically 10-20 (if much higher/lower, might indicate issues)
- `sfreq`: Should be 250.0 Hz

---

## Troubleshooting

### Problem: "dyconnmap not found" or import errors

**Solution**: This script does NOT use dyconnmap (I implemented MVAR/DTF/PDC from scratch to avoid dependency issues). Just make sure you have `numpy`, `scipy`, `matplotlib` installed.

### Problem: "MVAR fitting failed"

**Causes**:
1. Data has NaN or Inf values
2. Data is constant (no variance)
3. Too few timepoints for model order

**Solution**: Check your preprocessing output. Make sure epochs have variation.

### Problem: Script is very slow

**Expected**: ~30-60 seconds per patient (depends on number of epochs)

**If it's slower**:
- Check how many epochs you have (files with 1000+ epochs will take longer)
- Consider using `--model_order 15` to fix order instead of auto-selection

### Problem: "All epochs failed"

**Check**:
1. Is your epochs file correct shape? Should be `(n_epochs, 22, 500)`
2. Run this test:
   ```python
   import numpy as np
   epochs = np.load("your_epochs.npy")
   print(f"Shape: {epochs.shape}")
   print(f"Has NaN: {np.any(np.isnan(epochs))}")
   print(f"Has Inf: {np.any(np.isinf(epochs))}")
   print(f"Min: {epochs.min()}, Max: {epochs.max()}")
   ```

### Problem: Visualization looks weird (all black or all white)

**All black**: No connectivity detected
- Check if preprocessing removed too much data
- Try lowering rejection threshold in preprocessing

**All white/random**: Model might be overfitting
- Try reducing model order: `--model_order 10`

---

## Next Steps (After Connectivity Works)

### Immediate Validation

1. **Visual inspection**: Open a few `*_visualization.png` files. Do epilepsy patients look different from controls?

2. **Statistical check**: Create a simple comparison script:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Load one epilepsy patient
   epi_dtf = np.load("connectivity_results/00_epilepsy/patient1_dtf_alpha.npy")
   
   # Load one control patient
   ctrl_dtf = np.load("connectivity_results/01_no_epilepsy/patient2_dtf_alpha.npy")
   
   # Compare average connectivity strength
   epi_mean = epi_dtf.mean(axis=0)  # Average over epochs
   ctrl_mean = ctrl_dtf.mean(axis=0)
   
   # Plot side by side
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   axes[0].imshow(epi_mean, cmap='hot', vmin=0, vmax=1)
   axes[0].set_title("Epilepsy Patient")
   axes[1].imshow(ctrl_mean, cmap='hot', vmin=0, vmax=1)
   axes[1].set_title("Control Patient")
   plt.savefig("connectivity_comparison.png")
   ```

### Moving to Step 3: Graph Construction

Once connectivity works, you'll need to:
1. Convert connectivity matrices → graph objects
2. Add node features (channel positions, band powers)
3. Threshold weak connections
4. Create PyTorch Geometric Data objects

**I can help you with this next!** Just confirm your connectivity analysis is working first.

---

## Quick Reference Commands

### Test single file
```bash
python connectivity_analysis.py \
  --epoch_file "path/to/patient_epochs.npy" \
  --output_dir "connectivity_results/test"
```

### Process all files (with limit for testing)
```bash
python connectivity_batch.py \
  --data_dir "data_pp" \
  --output_dir "connectivity_results" \
  --max_files 10
```

### Process all files (full run)
```bash
python connectivity_batch.py \
  --data_dir "data_pp" \
  --output_dir "connectivity_results" \
  --skip_existing
```

### Fix model order (if auto-selection is unstable)
```bash
python connectivity_analysis.py \
  --epoch_file "path/to/patient_epochs.npy" \
  --output_dir "connectivity_results" \
  --model_order 15
```

---

## Questions?

If something doesn't work:
1. Check the error message carefully
2. Look at the Troubleshooting section
3. Verify your input data shape: `(n_epochs, 22, 500)`
4. Try with a smaller test file first

**Remember**: The goal is to get connectivity matrices that you can turn into graphs. Once this step works, the next steps (graph construction and GNN) will be much easier!
