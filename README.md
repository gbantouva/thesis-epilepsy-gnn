# Thesis: Graph-Based Self-Supervised Learning for Epilepsy Detection

## Repo structure
- `data_raw/`   → raw TUEP EDFs (ignored by git)
- `data_pp/`   → preprocessed numpy/pkl outputs (ignored by git)
- `figures/`   → plots (ignored by git)
- `src/`       → Python scripts
- `notebooks/` → Jupyter notebooks for exploratory analysis

## Preprocessing a single EDF

Usage:
```bash
python src/preprocess_single.py \
  --edf "data_raw/DATA/01_no_epilepsy/aaaaafiy/s002_2010/01_tcp_ar/aaaaafiy_s002_t001.edf" \
  --out "data_pp" \
  --psd_dir "figures/psd"
