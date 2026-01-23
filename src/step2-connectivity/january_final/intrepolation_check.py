import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths and channels
pp_dir = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced")

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2']

# Initialize counters
epilepsy_interp = {ch: 0 for ch in CHANNEL_NAMES}
control_interp = {ch: 0 for ch in CHANNEL_NAMES}
epilepsy_count = 0
control_count = 0

print("Scanning files...")

# 1. Process Files
for mask_file in pp_dir.rglob("*_present_mask.npy"):
    try:
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
            
    except Exception as e:
        print(f"Error reading {mask_file.name}: {e}")

# Avoid division by zero if folder is empty
if epilepsy_count == 0: epilepsy_count = 1
if control_count == 0: control_count = 1

# 2. Calculate Rates for Visualization
epi_rates = [epilepsy_interp[ch] / epilepsy_count * 100 for ch in CHANNEL_NAMES]
con_rates = [control_interp[ch] / control_count * 100 for ch in CHANNEL_NAMES]
diffs = [e - c for e, c in zip(epi_rates, con_rates)]

# 3. Print Text Report
print("\nChannel Interpolation Rates:")
print(f"{'Channel':<8} {'Epilepsy':<12} {'Control':<12} {'Difference':<12}")
print("-" * 44)
for i, ch in enumerate(CHANNEL_NAMES):
    flag = "⚠️" if abs(diffs[i]) > 5 else ""
    print(f"{ch:<8} {epi_rates[i]:>8.1f}%    {con_rates[i]:>8.1f}%    {diffs[i]:>+8.1f}% {flag}")

# 4. Generate Visualization
x = np.arange(len(CHANNEL_NAMES))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width/2, epi_rates, width, label='Epilepsy', color='#ff7f0e', alpha=0.8)
rects2 = ax.bar(x + width/2, con_rates, width, label='Control', color='#1f77b4', alpha=0.8)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Interpolation Rate (%)')
ax.set_title('Interpolation Rates by Channel and Group')
ax.set_xticks(x)
ax.set_xticklabels(CHANNEL_NAMES)
ax.legend()

# Add a threshold line at 5% difference (visual guide only)
# To make it cleaner, we add grid lines on Y-axis
ax.grid(axis='y', linestyle='--', alpha=0.5)

fig.tight_layout()
plt.show()