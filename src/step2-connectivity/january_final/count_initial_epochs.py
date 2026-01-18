import numpy as np
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# ΠΡΟΣΟΧΗ: Βάλε εδώ τον φάκελο με τα PREPROCESSED δεδομένα σου (τα .npy)
# =============================================================================
# Π.χ. F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced
INPUT_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced")

print(f"🔍 Scanning input directory: {INPUT_DIR}")

# Βρες όλα τα αρχεία _epochs.npy
files = list(INPUT_DIR.rglob("*_epochs.npy"))

if not files:
    print("❌ Δεν βρέθηκαν αρχεία _epochs.npy! Έλεγξε το path.")
    exit()

total_epochs = 0
epilepsy_epochs = 0
control_epochs = 0

print(f"📂 Found {len(files)} files. Counting initial epochs...")

for f in tqdm(files):
    try:
        # mmap_mode='r' διαβάζει μόνο τα metadata (μέγεθος) χωρίς να φορτώσει τα GB στη μνήμη!
        # Είναι ακαριαίο.
        data = np.load(f, mmap_mode='r')
        
        n = len(data)  # Το πρώτο dimension είναι τα epochs
        total_epochs += n
        
        # Καταμέτρηση ανά ομάδα
        if '00_epilepsy' in str(f):
            epilepsy_epochs += n
        elif '01_no_epilepsy' in str(f):
            control_epochs += n
            
    except Exception as e:
        print(f"⚠️ Error reading {f.name}: {e}")

print("\n" + "="*50)
print("📊 INITIAL EPOCH COUNT (PREPROCESSED)")
print("="*50)
print(f"Files Scanned:    {len(files)}")
print(f"TOTAL Epochs:     {total_epochs:,}")
print("-" * 30)
print(f"🔴 Epilepsy:      {epilepsy_epochs:,}")
print(f"🔵 Control:       {control_epochs:,}")
print("="*50)