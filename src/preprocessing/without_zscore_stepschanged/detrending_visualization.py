"""
Detrending Visualization Functions
===================================

Add these functions to your preprocess_single.py or create a separate module.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def plot_detrending_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    channel_names: List[str],
    sampling_rate: float,
    epoch_duration: float,
    patient_id: str,
    output_dir: Path,
    channels_to_plot: List[str] = None,
    time_window_start: float = 0.0,
    time_window_duration: float = 20.0
):
    """
    Create comparison plots showing epochs before and after detrending.
    
    Args:
        data_before: Epochs data BEFORE detrending (n_epochs, n_channels, n_samples)
        data_after: Epochs data AFTER detrending (n_epochs, n_channels, n_samples)
        channel_names: List of channel names (22 CORE_CHS)
        sampling_rate: Sampling rate in Hz
        epoch_duration: Duration of each epoch in seconds
        patient_id: Patient identifier for plot title
        output_dir: Where to save the plot
        channels_to_plot: Subset of channels to visualize (default: first 6)
        time_window_start: Start time in seconds (default: 0)
        time_window_duration: Duration to plot in seconds (default: 20)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select channels to plot
    if channels_to_plot is None:
        # Default: plot first 6 channels
        channels_to_plot = channel_names[:6]
    
    # Get channel indices
    ch_indices = [channel_names.index(ch) for ch in channels_to_plot if ch in channel_names]
    
    if not ch_indices:
        print("⚠ No valid channels to plot")
        return
    
    # Calculate which epochs to include in time window
    start_epoch = int(time_window_start / epoch_duration)
    n_epochs_in_window = int(np.ceil(time_window_duration / epoch_duration))
    end_epoch = start_epoch + n_epochs_in_window
    
    # Clip to available epochs
    n_total_epochs = data_before.shape[0]
    end_epoch = min(end_epoch, n_total_epochs)
    
    if start_epoch >= n_total_epochs:
        print(f"⚠ Time window start ({time_window_start}s) exceeds data length")
        return
    
    # Concatenate epochs to create continuous signal
    epochs_to_plot = slice(start_epoch, end_epoch)
    
    signal_before = np.concatenate([data_before[i, :, :] for i in range(start_epoch, end_epoch)], axis=1)
    signal_after = np.concatenate([data_after[i, :, :] for i in range(start_epoch, end_epoch)], axis=1)
    
    # Create time vector
    n_samples = signal_before.shape[1]
    times = np.arange(n_samples) / sampling_rate + (start_epoch * epoch_duration)
    
    # Create figure
    n_channels = len(ch_indices)
    fig, axes = plt.subplots(n_channels, 2, figsize=(16, 3*n_channels))
    
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        f'Per-Epoch Detrending Comparison\n{patient_id}\n'
        f'(After: Resampling → Filtering → CAR → Epoching)',
        fontsize=14, fontweight='bold'
    )
    
    for idx, ch_idx in enumerate(ch_indices):
        ch_name = channel_names[ch_idx]
        
        # Convert to microvolts
        before_uv = signal_before[ch_idx, :] * 1e6
        after_uv = signal_after[ch_idx, :] * 1e6
        
        # Plot BEFORE detrending
        axes[idx, 0].plot(times, before_uv, 'b-', linewidth=0.5)
        axes[idx, 0].set_ylabel(f'{ch_name}\n(µV)', fontweight='bold')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Add mean line
        mean_before = np.mean(before_uv)
        axes[idx, 0].axhline(mean_before, color='red', linestyle='--', 
                            alpha=0.5, linewidth=1.5, label=f'Mean={mean_before:.2f}')
        
        if idx == 0:
            axes[idx, 0].set_title('BEFORE Detrending', fontweight='bold', fontsize=12)
        
        if idx == n_channels - 1:
            axes[idx, 0].set_xlabel('Time (s)')
        
        axes[idx, 0].legend(loc='upper right', fontsize=8)
        
        # Add epoch boundaries
        for epoch_boundary in np.arange(times[0], times[-1], epoch_duration):
            axes[idx, 0].axvline(epoch_boundary, color='gray', linestyle=':', 
                                alpha=0.3, linewidth=0.8)
        
        # Plot AFTER detrending
        axes[idx, 1].plot(times, after_uv, 'g-', linewidth=0.5)
        axes[idx, 1].set_ylabel(f'{ch_name}\n(µV)', fontweight='bold')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Add mean line
        mean_after = np.mean(after_uv)
        axes[idx, 1].axhline(mean_after, color='red', linestyle='--', 
                            alpha=0.5, linewidth=1.5, label=f'Mean={mean_after:.2f}')
        
        if idx == 0:
            axes[idx, 1].set_title('AFTER Per-Epoch Detrending', fontweight='bold', fontsize=12)
        
        if idx == n_channels - 1:
            axes[idx, 1].set_xlabel('Time (s)')
        
        axes[idx, 1].legend(loc='upper right', fontsize=8)
        
        # Add epoch boundaries
        for epoch_boundary in np.arange(times[0], times[-1], epoch_duration):
            axes[idx, 1].axvline(epoch_boundary, color='orange', linestyle='--', 
                                alpha=0.5, linewidth=1.2)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{patient_id}_detrending_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved detrending comparison: {output_file.name}")


def plot_trend_analysis(
    data_before: np.ndarray,
    channel_names: List[str],
    sampling_rate: float,
    epoch_duration: float,
    patient_id: str,
    output_dir: Path,
    channels_to_plot: List[str] = None,
    time_window_start: float = 0.0,
    time_window_duration: float = 20.0
):
    """
    Create plots showing the per-epoch trends that will be removed.
    
    Shows the linear trend for each epoch with slopes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select channels
    if channels_to_plot is None:
        channels_to_plot = channel_names[:6]
    
    ch_indices = [channel_names.index(ch) for ch in channels_to_plot if ch in channel_names]
    
    if not ch_indices:
        return
    
    # Calculate epochs in window
    start_epoch = int(time_window_start / epoch_duration)
    n_epochs_in_window = min(5, data_before.shape[0] - start_epoch)  # Max 5 epochs
    
    if n_epochs_in_window <= 0:
        return
    
    # Create figure
    n_channels = len(ch_indices)
    fig, axes = plt.subplots(n_channels, n_epochs_in_window, 
                            figsize=(4*n_epochs_in_window, 3*n_channels))
    
    if n_channels == 1:
        axes = axes.reshape(1, -1)
    if n_epochs_in_window == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(
        f'Per-Epoch Linear Trend Analysis\n{patient_id}',
        fontsize=14, fontweight='bold'
    )
    
    for ch_idx_pos, ch_idx in enumerate(ch_indices):
        ch_name = channel_names[ch_idx]
        
        for epoch_offset in range(n_epochs_in_window):
            epoch_num = start_epoch + epoch_offset
            
            # Get epoch data
            epoch_data = data_before[epoch_num, ch_idx, :] * 1e6  # Convert to µV
            
            # Create time vector for this epoch
            epoch_times = np.arange(len(epoch_data)) / sampling_rate
            
            # Fit linear trend
            coeffs = np.polyfit(epoch_times, epoch_data, 1)
            trend = np.polyval(coeffs, epoch_times)
            
            # Plot
            ax = axes[ch_idx_pos, epoch_offset]
            ax.plot(epoch_times, epoch_data, 'b-', linewidth=1, alpha=0.7, 
                   label='Signal')
            ax.plot(epoch_times, trend, 'r-', linewidth=2, 
                   label=f'Trend (slope={coeffs[0]:.3f})')
            ax.fill_between(epoch_times, epoch_data, trend, alpha=0.2, color='yellow')
            
            # Assess slope magnitude
            slope_mag = abs(coeffs[0])
            if slope_mag < 0.1:
                assessment = "Small"
                color = 'green'
            elif slope_mag < 1.0:
                assessment = "Moderate"
                color = 'orange'
            else:
                assessment = "Large"
                color = 'red'
            
            ax.set_title(
                f'Epoch {epoch_num}\nSlope: {coeffs[0]:.3f} µV/s ({assessment})',
                fontsize=9, color=color
            )
            
            if epoch_offset == 0:
                ax.set_ylabel(f'{ch_name}\n(µV)', fontweight='bold')
            
            if ch_idx_pos == n_channels - 1:
                ax.set_xlabel('Time (s)')
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=7)
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{patient_id}_trend_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved trend analysis: {output_file.name}")