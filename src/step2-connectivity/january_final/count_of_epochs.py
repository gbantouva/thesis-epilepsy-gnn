import numpy as np
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# ΡΥΘΜΙΣΗ: Βάλε τον φάκελο με τα αποτελέσματα του Step 2 (τα .npz)
# =============================================================================
RESULTS_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_fixed_15")

print(f"🔍 Scanning directory: {RESULTS_DIR}")

# Βρες όλα τα αρχεία _graphs.npz
files = list(RESULTS_DIR.rglob("*_graphs.npz"))

if not files:
    print("❌ Δεν βρέθηκαν αρχεία .npz! Έλεγξε το path.")
    exit()

total_epochs = 0
epilepsy_epochs = 0
control_epochs = 0
files_count = 0

print(f"📂 Found {len(files)} files. Counting epochs...")

for f in tqdm(files):
    try:
        # Φόρτωσε το αρχείο (χωρίς να φορτώσεις όλο το data στη μνήμη, μόνο τα keys)
        with np.load(f) as data:
            # Το πλήθος των epochs είναι το μήκος του πίνακα 'orders' (ή 'indices')
            n = len(data['orders'])
            
            total_epochs += n
            files_count += 1
            
            if '00_epilepsy' in str(f):
                epilepsy_epochs += n
            else:
                control_epochs += n
                
    except Exception as e:
        print(f"⚠️ Error reading {f.name}: {e}")

print("\n" + "="*50)
print("📊 FINAL EPOCH COUNT (STABLE ONLY)")
print("="*50)
print(f"Files Processed:  {files_count}")
print(f"TOTAL Epochs:     {total_epochs:,}")
print("-" * 30)
print(f"🔴 Epilepsy:      {epilepsy_epochs:,}")
print(f"🔵 Control:       {control_epochs:,}")
print("="*50)