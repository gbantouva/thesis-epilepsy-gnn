# Connectivity Analysis - Quick Reference & Progress Tracker

## 🎯 Your Mission

Convert preprocessed EEG data → Connectivity matrices (DTF/PDC) → Ready for graph construction

---

## 📋 Step-by-Step Checklist

### Phase 1: Setup (10 minutes)
```
[ ] Downloaded all 6 Python scripts from Claude
[ ] Copied them to: F:\October-Thesis\thesis-epilepsy-gnn\connectivity_scripts\
[ ] Installed packages: pip install numpy scipy matplotlib tqdm seaborn mne
[ ] Verified Python version: python --version (should be 3.10-3.12)
```

### Phase 2: Test Run (3 minutes)
```
[ ] Ran: python test_connectivity_pipeline.py --data_dir <your_data_pp> --output_dir connectivity_test
[ ] Waited ~2 minutes for completion
[ ] Checked output folder exists: connectivity_test/
[ ] Opened: connectivity_test/test_comparison.png
[ ] Verified: Can see patterns (red/blue colors, not all grey)
```

**⚠️ STOP HERE if test fails** - Check troubleshooting section!

### Phase 3: Full Processing (1-2 hours)
```
[ ] Ran: python connectivity_batch.py --data_dir <your_data_pp> --output_dir connectivity_results --skip_existing
[ ] Checked progress bar is running
[ ] Waited for completion (status: X/Y successful)
[ ] Success rate is >80%
[ ] Output folder has subdirectories: 00_epilepsy/ and 01_no_epilepsy/
```

### Phase 4: Validation (5 minutes)
```
[ ] Ran: python validate_connectivity.py --connectivity_dir connectivity_results
[ ] Read validation_report.txt
[ ] Most patients show status: "OK"
[ ] Opened quality_metrics.png
[ ] Distributions look reasonable (bell-shaped, not all zeros)
```

### Phase 5: Group Comparison (5 minutes)
```
[ ] Ran: python compare_groups.py --connectivity_dir connectivity_results --output_dir comparison_results
[ ] Opened: comparison_results/global_connectivity_comparison.png
[ ] Some bands show statistical significance (* or ** or ***)
[ ] Difference plots show structure (not uniform)
[ ] Have figures ready for thesis!
```

---

## 🚦 Status Indicators

### ✅ Everything is Working
- Test pipeline completes without errors
- Success rate >80% in batch processing
- Validation shows "Passed" for most patients
- Visualizations show clear patterns
- Group comparison shows significant differences

### ⚠️ Warning Signs (But Might Be OK)
- Success rate 60-80% (some problematic patients)
- A few warnings in validation report
- Some bands show no significant difference
- Model orders vary widely (5-25)

### ❌ Critical Issues (Need Fixing)
- Test pipeline fails completely
- Success rate <50%
- Most patients show "FAILED" in validation
- All visualizations are black/white/uniform
- NaN or Inf errors
- Import errors (missing packages)

---

## 🎨 What Good Results Look Like

### Connectivity Visualization
```
HOT COLORS (red/yellow) = Strong connections
- Should see some hot spots (not all cold)
- Should NOT be all hot (not all 1.0)
- Pattern should vary across frequency bands
- Diagonal should be relatively low
```

### Group Comparison
```
DIFFERENCE PLOT (Epilepsy - Control)
- Red = Epilepsy has stronger connections
- Blue = Control has stronger connections
- Should see organized patterns (not random)
- Statistical markers (* ** ***) indicate significance
```

### Typical Values
```
DTF Mean: 0.2 - 0.4
PDC Mean: 0.1 - 0.3
MVAR Order: 10 - 20
Diagonal (self-connections): <0.3
```

---

## 🔍 Quick Diagnostic Commands

### Check if preprocessing data is OK
```python
import numpy as np
epochs = np.load("path/to/patient_epochs.npy")
print(f"Shape: {epochs.shape}")  # Should be (n, 22, 500)
print(f"NaN: {np.any(np.isnan(epochs))}")  # Should be False
print(f"Range: [{epochs.min():.2f}, {epochs.max():.2f}]")
print(f"Std: {epochs.std():.4f}")  # Should be >0.01
```

### Check if connectivity results exist
```bash
# Windows
dir connectivity_results\00_epilepsy /s

# Linux/Mac
find connectivity_results/00_epilepsy -name "*.npy"
```

### Count successful patients
```bash
# Windows
dir connectivity_results /s /b | find /c "_dtf_alpha.npy"

# Linux/Mac
find connectivity_results -name "*_dtf_alpha.npy" | wc -l
```

### Load and inspect connectivity
```python
import numpy as np
dtf = np.load("connectivity_results/.../patient_dtf_alpha.npy")
print(f"Shape: {dtf.shape}")  # (n_epochs, 22, 22)
print(f"Mean: {dtf.mean():.4f}")
print(f"Std: {dtf.std():.4f}")
print(f"Range: [{dtf.min():.4f}, {dtf.max():.4f}]")

# Check diagonal (self-connections)
import numpy as np
diag_mean = np.array([dtf[i].diagonal().mean() for i in range(len(dtf))])
print(f"Diagonal mean: {diag_mean.mean():.4f}")  # Should be <0.5
```

---

## 📞 Emergency Troubleshooting

### Script crashes immediately
```
1. Check Python version: python --version
2. Reinstall packages: pip install --upgrade numpy scipy matplotlib mne
3. Try Python 3.10: conda create -n thesis python=3.10
4. Check file paths (use absolute paths if relative don't work)
```

### "All epochs failed"
```
1. Check epochs file: epochs.shape should be (n, 22, 500)
2. Check for NaN: np.any(np.isnan(epochs)) should be False
3. Check variance: epochs.std() should be >0.01
4. Try different patient file
```

### Very slow processing
```
1. Limit test: --max_files 5
2. Fix MVAR order: --model_order 15
3. Check epoch count: fewer epochs = faster
4. Check CPU usage (should be high during processing)
```

### Results look weird
```
1. Visual inspection: Open PNG files
2. Check value ranges: Load .npy, check min/max
3. Compare with different patient
4. Re-run with fixed MVAR order: --model_order 12
```

---

## 🎓 For Your Thesis Defense

### Questions Professors Might Ask

**Q: "Why DTF and PDC instead of just correlation?"**  
A: "DTF and PDC capture directed (causal) relationships and are frequency-resolved, revealing how information flows between brain regions. Simple correlation is symmetric and doesn't capture directionality."

**Q: "How did you choose the MVAR model order?"**  
A: "I used the Akaike Information Criterion (AIC) for automatic model order selection, which balances model complexity against goodness of fit. Typical orders were 10-20 for 2-second epochs at 250 Hz."

**Q: "Do epilepsy and control groups show different connectivity?"**  
A: "Yes, statistical tests revealed significant differences [point to specific bands with p<0.05]. Epilepsy patients showed [increased/decreased] connectivity in [theta/alpha/etc] bands, consistent with literature on epileptic networks."

**Q: "Why these frequency bands?"**  
A: "These are standard clinical EEG bands: delta (deep sleep/pathology), theta (drowsiness), alpha (relaxation), beta (active thinking), gamma (information processing). Epilepsy research shows band-specific alterations in brain connectivity."

---

## 📊 Files for Your Thesis

### Essential Figures
1. `test_comparison.png` - Example of connectivity computation
2. `*_connectivity_visualization.png` - Per-patient connectivity matrices
3. `global_connectivity_comparison.png` - Main results figure (Epilepsy vs Control)
4. `dtf_alpha_comparison.png` - Detailed alpha band comparison

### Essential Data
1. `connectivity_batch_summary.json` - Processing statistics
2. `validation_report.txt` - Quality control results
3. `*_dtf_*.npy` - Connectivity matrices (for graph construction)
4. `*_connectivity_metadata.json` - Patient-level metadata

### Text Snippets for Methods Section
```
"Effective connectivity was computed using multivariate autoregressive (MVAR) 
models fitted to 2-second EEG epochs. Model order was selected using the Akaike 
Information Criterion. We computed Directed Transfer Function (DTF) and Partial 
Directed Coherence (PDC) in five frequency bands: delta (0.5-4 Hz), theta 
(4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), and gamma (30-100 Hz). This 
resulted in 22×22 connectivity matrices representing directed information 
flow between all channel pairs."
```

---

## 🎯 Next: Graph Construction

Once all checkboxes above are ✅, you're ready for:

**Step 3: Graph Construction**
- Convert connectivity matrices → PyTorch Geometric graphs
- Add node features (channel positions, band powers)
- Create train/val/test splits
- Prepare for GNN training

**I can create scripts for this next step!** Just confirm connectivity works first.

---

## 📞 Quick Help

| Problem | Solution |
|---------|----------|
| Import errors | `pip install numpy scipy matplotlib mne` |
| Slow processing | Use `--max_files 5` for testing |
| All black visualizations | Check preprocessing, try different patient |
| NaN/Inf errors | Verify epochs data: `np.isnan(epochs).any()` |
| Can't find patients | Check folder names: 00_epilepsy, 01_no_epilepsy |
| Crashes after 1-2 patients | Possible memory issue, process in smaller batches |

---

**🚀 You got this! Follow the checklist step by step, and you'll have connectivity results in no time!**
