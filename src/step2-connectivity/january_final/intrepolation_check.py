import numpy as np
from pathlib import Path

# Check interpolation rates per group
pp_dir = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced")

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']

epilepsy_interp = {ch: 0 for ch in CHANNEL_NAMES}
control_interp = {ch: 0 for ch in CHANNEL_NAMES}
epilepsy_count = 0
control_count = 0

for mask_file in pp_dir.rglob("*_present_mask.npy"):
    mask = np.load(mask_file)  # True = present, False = interpolated
    is_epilepsy = '00_epilepsy' in str(mask_file)
    
    for i, ch in enumerate(CHANNEL_NAMES):
        if not mask[i]:  # Channel was interpolated
            if is_epilepsy:
                epilepsy_interp[ch] += 1
            else:
                control_interp[ch] += 1
    
    if is_epilepsy:
        epilepsy_count += 1
    else:
        control_count += 1

print("Channel Interpolation Rates:")
print(f"{'Channel':<8} {'Epilepsy':<12} {'Control':<12} {'Difference':<12}")
print("-" * 44)
for ch in CHANNEL_NAMES:
    epi_rate = epilepsy_interp[ch] / epilepsy_count * 100
    con_rate = control_interp[ch] / control_count * 100
    diff = epi_rate - con_rate
    flag = "⚠️" if abs(diff) > 5 else ""
    print(f"{ch:<8} {epi_rate:>8.1f}%    {con_rate:>8.1f}%    {diff:>+8.1f}% {flag}")