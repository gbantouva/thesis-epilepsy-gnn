import mne

# Load the EDF file (header only, no need to load data)
raw = mne.io.read_raw_edf(r'F:\October-Thesis\thesis-epilepsy-gnn\data_raw\DATA\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001.edf', preload=False)

# Get duration in seconds
duration_seconds = raw.times[-1]

# Or get it directly
duration_seconds1 = raw.n_times / raw.info['sfreq']

print(f"Duration: {duration_seconds} seconds")
print(f"Duration: {duration_seconds/60:.2f} minutes")
print(f"Duration: {duration_seconds/3600:.2f} hours")

print(f"Duration: {duration_seconds1} seconds")
print(f"Duration: {duration_seconds1/60:.2f} minutes")
print(f"Duration: {duration_seconds1/3600:.2f} hours")