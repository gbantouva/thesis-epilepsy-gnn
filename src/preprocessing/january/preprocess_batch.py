"""
Batch EDF file preprocessing script.

Usage:
  python src/preprocess_batch.py --input_dir data_raw/DATA --output_dir data_pp --psd_dir figures/psd
  
Example with limits:
  python src/preprocess_batch.py --input_dir data_raw/DATA --output_dir data_pp --psd_dir figures/psd --max_files 50
  
This script:
1. Recursively finds all .edf files in input directory
2. Preprocesses each file using the optimized pipeline
3. Saves results to output directories
4. Generates a summary report
"""

import sys
from pathlib import Path

# Allow importing from src/ directory
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import json
import time
from datetime import datetime
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from preprocess_core import preprocess_single


def find_edf_files(input_dir: Path, max_files: int = None) -> List[Path]:
    """
    Recursively find all EDF files in directory.
    
    Args:
        input_dir: Root directory to search
        max_files: Maximum number of files to return (None = all)
        
    Returns:
        List of paths to EDF files
    """
    edf_files = sorted(input_dir.rglob("*.edf"))
    
    if max_files is not None:
        edf_files = edf_files[:max_files]
    
    return edf_files


def save_preprocessed_file(pid: str, res: Dict, out_dir: Path, psd_dir: Path, 
                           save_psd: bool = True, verbose: bool = True) -> bool:
    """
    Save preprocessing results for a single file.
    
    Args:
        pid: Patient ID (filename stem)
        res: Results dictionary from preprocess_single()
        out_dir: Output directory for arrays/metadata
        psd_dir: Output directory for PSD plots
        save_psd: Whether to save PSD plots
        verbose: Whether to print progress
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Save numpy arrays
        np.save(out_dir / f"{pid}_epochs.npy", res["epochs"].get_data())
        np.save(out_dir / f"{pid}_labels.npy", res["labels"])
        #np.save(out_dir / f"{pid}_raw.npy", res["raw_after"].get_data())
        np.save(out_dir / f"{pid}_present_mask.npy", res["present_mask"])
        
        # Save metadata
        with open(out_dir / f"{pid}_info.pkl", "wb") as f:
            pickle.dump(res["raw_after"].info, f)
        
        with open(out_dir / f"{pid}_present_channels.json", "w", encoding="utf-8") as f:
            json.dump(res["present_channels"], f, ensure_ascii=False, indent=2)
        
        # Save PSD plots
        if save_psd:
            for tag, psd in [("before", res.get("psd_before")), 
                            ("after", res.get("psd_after"))]:
                if psd is None:
                    continue
                
                try:
                    fig = psd.plot(show=False, average='mean', spatial_colors=False)
                    fig.suptitle(f"{pid} - PSD {tag.upper()}")
                    fig.savefig(psd_dir / f"{pid}_PSD_{tag}.png", dpi=100, bbox_inches="tight")
                    plt.close(fig)
                except Exception as e:
                    if verbose:
                        print(f"    ⚠ Could not save PSD {tag}: {e}")
                    plt.close('all')
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"    ✗ Failed to save {pid}: {e}")
        return False


def main():
    """
    Main function for batch preprocessing.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Batch preprocess EEG EDF files for epilepsy detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in directory
  python src/preprocess_batch.py --input_dir data_raw/DATA --output_dir data_pp --psd_dir figures/psd
  
  # Process first 50 files only
  python src/preprocess_batch.py --input_dir data_raw/DATA --output_dir data_pp --psd_dir figures/psd --max_files 50
  
  # Custom parameters
  python src/preprocess_batch.py --input_dir data --output_dir output --psd_dir figs --notch 50 --band 1 100
        """
    )
    
    parser.add_argument(
        "--input_dir", 
        required=True, 
        help="Input directory containing EDF files (searched recursively)"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Output directory for preprocessed arrays and metadata"
    )
    parser.add_argument(
        "--psd_dir", 
        required=True, 
        help="Output directory for PSD figures"
    )
    parser.add_argument(
        "--max_files", 
        type=int, 
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--notch", 
        type=float, 
        default=60.0, 
        help="Notch filter frequency in Hz (default: 60.0)"
    )
    parser.add_argument(
        "--band", 
        type=float, 
        nargs=2, 
        default=[0.5, 80.0],
        metavar=('LOW', 'HIGH'),
        help="Bandpass filter frequencies in Hz (default: 0.5 80.0)"
    )
    parser.add_argument(
        "--resample", 
        type=float, 
        default=250.0,
        help="Target sampling rate in Hz (default: 250.0)"
    )
    parser.add_argument(
        "--epoch_len", 
        type=float, 
        default=4.0,
        help="Epoch duration in seconds (default: 4.0)"
    )
    parser.add_argument(
        "--epoch_overlap", 
        type=float, 
        default=0.0,
        help="Overlap between epochs in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--reject_percentile", 
        type=float, 
        default=95.0,
        help="Percentile for adaptive rejection threshold (default: 95.0)"
    )
    #parser.add_argument(
    #    "--reject_cap", 
    #    type=float, 
    #    default=500.0,
    #    help="Hard cap for rejection threshold in µV (default: 500.0)"
    #)
    parser.add_argument(
        "--no_psd", 
        action="store_true",
        help="Skip PSD computation and plotting"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress per-file progress messages (still shows summary)"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    psd_dir = Path(args.psd_dir)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_psd:
        psd_dir.mkdir(parents=True, exist_ok=True)
    
    # Find EDF files
    print("\nSearching for EDF files...")
    edf_files = find_edf_files(input_dir, args.max_files)
    
    if len(edf_files) == 0:
        print(f"Error: No EDF files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(edf_files)} EDF files")
    
    # Print configuration
    print("\n" + "="*70)
    print("EEG BATCH PREPROCESSING")
    print("="*70)
    print(f"Input directory:     {input_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"PSD directory:       {psd_dir}")
    print(f"Files to process:    {len(edf_files)}")
    print(f"Notch filter:        {args.notch} Hz")
    print(f"Bandpass filter:     {args.band[0]}-{args.band[1]} Hz")
    print(f"Resample rate:       {args.resample} Hz")
    print(f"Epoch length:        {args.epoch_len} s")
    print(f"Epoch overlap:       {args.epoch_overlap} s")
    print(f"Reject percentile:   {args.reject_percentile}th")
    #print(f"Reject cap:          {args.reject_cap} µV")
    print("="*70 + "\n")
    
    # Initialize statistics
    stats = {
        "total_files_found": len(edf_files),
        "total_files_processed": 0,
        "total_files_skipped": 0,  # <-- ADD THIS
        "successful": 0,
        "failed": 0,
        "epilepsy_count": 0,
        "control_count": 0,
        "total_epochs": 0,
        "total_interpolated": 0,
        "processing_times": [],
        "failed_files": []
    }
    
    start_time = time.time()
    
    # Process each file
    for edf_path in tqdm(edf_files, desc="Processing files", disable=args.quiet):
        pid = edf_path.stem
        
        # --- START: SKIP LOGIC ---
        # Find the *relative* path to preserve folder structure
        try:
            relative_parent = edf_path.parent.relative_to(input_dir)
        except ValueError:
            relative_parent = Path() # File is in root of input_dir
            
        final_out_dir = output_dir / relative_parent
        final_psd_dir = psd_dir / relative_parent
        expected_output_file = final_out_dir / f"{pid}_epochs.npy"
        
        if expected_output_file.exists():
            stats["total_files_skipped"] += 1
            continue  # <-- This is the skip
        # --- END: SKIP LOGIC ---
        
        # Create the subdirectories *just in time*
        final_out_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_psd:
            final_psd_dir.mkdir(parents=True, exist_ok=True)

        stats["total_files_processed"] += 1
        file_start = time.time()
        
        # --- START OF NEW TRY/EXCEPT BLOCK ---
        try:
            # Preprocess
            res = preprocess_single(
                edf_path,
                notch=args.notch,
                band=tuple(args.band),
                resample_hz=args.resample,
                epoch_len=args.epoch_len,
                epoch_overlap=args.epoch_overlap,
                reject_percentile=args.reject_percentile,
                # reject_cap_uv=args.reject_cap, # This was correctly removed
                return_psd=not args.no_psd,
                verbose=False # TQDM handles progress
            )
                
            file_time = time.time() - file_start
            stats["processing_times"].append(file_time)
                
            # Check if preprocessing failed
            if res is None:
                # This handles "soft fails" from within preprocess_single
                raise ValueError("preprocess_single returned None")
                
            # Save results (Note: pass final_out_dir and final_psd_dir)
            success = save_preprocessed_file(
                pid, res, final_out_dir, final_psd_dir, 
                save_psd=not args.no_psd, 
                verbose=not args.quiet
            )
                
            if success:
                stats["successful"] += 1
                stats["total_epochs"] += len(res["epochs"])
                #stats["total_interpolated"] += (~res["present_mask"]).sum()
                stats["total_interpolated"] += int((~res["present_mask"]).sum())

                if res["labels"][0] == 1:
                    stats["epilepsy_count"] += 1
                else:
                    stats["control_count"] += 1
                    
                if not args.quiet:
                    tqdm.write(f" ✓ Saved {pid}: {len(res['epochs'])} epochs ({file_time:.1f}s)")
            else:
                raise Exception("Save failed") # Will be caught by the 'except'
                
        except Exception as e:
            # This is the "catch-all" for this file
            file_time = time.time() - file_start
            stats["processing_times"].append(file_time)
            stats["failed"] += 1
            stats["failed_files"].append(str(edf_path))
            # Log the error to the console without crashing
            tqdm.write(f" ✗ FAILED {pid} ({file_time:.1f}s): {e}")
        # --- END OF NEW TRY/EXCEPT BLOCK ---
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PREPROCESSING SUMMARY")
    print("="*70)
    print(f"Total files found:     {stats['total_files_found']}")
    print(f"Total files skipped:   {stats['total_files_skipped']}")
    print(f"Total files processed: {stats['total_files_processed']}")
    #print(f"Successful:           {stats['successful']} ({100*stats['successful']/stats['total_files']:.1f}%)")
    #print(f"Failed:               {stats['failed']} ({100*stats['failed']/stats['total_files']:.1f}%)")
    # Calculate rates based on *processed* files, not total found
    success_rate = (stats['successful'] / max(stats['total_files_processed'], 1)) * 100
    fail_rate = (stats['failed'] / max(stats['total_files_processed'], 1)) * 100

    print(f"Successful:            {stats['successful']} ({success_rate:.1f}%)")
    print(f"Failed:                {stats['failed']} ({fail_rate:.1f}%)")
    print(f"")
    print(f"Epilepsy files:       {stats['epilepsy_count']}")
    print(f"Control files:        {stats['control_count']}")
    print(f"")
    print(f"Total epochs:         {stats['total_epochs']}")
    print(f"Avg epochs/file:      {stats['total_epochs']/max(stats['successful'],1):.1f}")
    print(f"Total interpolated:   {stats['total_interpolated']} channels")
    print(f"Avg interpolated/file: {stats['total_interpolated']/max(stats['successful'],1):.1f}")
    print(f"")
    print(f"Total time:           {total_time/60:.1f} minutes")
    print(f"Avg time/file:        {np.mean(stats['processing_times']):.1f} seconds")
    print(f"Output directory:     {output_dir}")
    print("="*70)
    
    # Save summary report
    summary_path = output_dir / "preprocessing_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "notch": args.notch,
            "band": args.band,
            "resample_hz": args.resample,
            "epoch_len": args.epoch_len,
            "epoch_overlap": args.epoch_overlap,
            "reject_percentile": args.reject_percentile
            #"reject_cap_uv": args.reject_cap
        },
        "statistics": {
            k: v for k, v in stats.items() if k != "processing_times"
        },
        "timing": {
            "total_seconds": total_time,
            "mean_seconds_per_file": float(np.mean(stats["processing_times"])),
            "std_seconds_per_file": float(np.std(stats["processing_times"]))
        }
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}\n")
    
    # Exit with error code if all files failed
    if stats["successful"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()