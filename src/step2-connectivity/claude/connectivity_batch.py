"""
Batch Connectivity Analysis - Process All Patients

This script finds all preprocessed epoch files and computes connectivity for each.

Usage:
  python connectivity_batch.py --data_dir data_pp --output_dir connectivity_results
  
Example:
  python connectivity_batch.py \
    --data_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\test\\data_pp \
    --output_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\connectivity_results \
    --max_files 10
"""

import sys
from pathlib import Path
import argparse
import time
from tqdm import tqdm
import json
import numpy as np

# Import the connectivity analysis module
from connectivity_analysis import process_file


def find_epoch_files(data_dir, max_files=None):
    """
    Find all *_epochs.npy files in directory.
    
    Args:
        data_dir: Root directory to search
        max_files: Maximum number of files to process (None = all)
    
    Returns:
        List of epoch file paths
    """
    data_dir = Path(data_dir)
    epoch_files = sorted(data_dir.rglob("*_epochs.npy"))
    
    if max_files is not None:
        epoch_files = epoch_files[:max_files]
    
    return epoch_files


def main():
    parser = argparse.ArgumentParser(
        description="Batch connectivity analysis for all patients",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory containing preprocessed data"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for connectivity results"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--model_order",
        type=int,
        default=None,
        help="Fixed MVAR model order (default: auto-select)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already have connectivity results"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all epoch files
    print("\nSearching for epoch files...")
    epoch_files = find_epoch_files(data_dir, args.max_files)
    
    if len(epoch_files) == 0:
        print(f"ERROR: No *_epochs.npy files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(epoch_files)} epoch files")
    
    # Initialize statistics
    stats = {
        "total_files": len(epoch_files),
        "processed": 0,
        "skipped": 0,
        "successful": 0,
        "failed": 0,
        "processing_times": [],
        "failed_files": []
    }
    
    # Print configuration
    print("\n" + "="*70)
    print("BATCH CONNECTIVITY ANALYSIS")
    print("="*70)
    print(f"Data directory:      {data_dir}")
    print(f"Output directory:    {output_dir}")
    print(f"Files to process:    {len(epoch_files)}")
    if args.model_order:
        print(f"MVAR model order:    {args.model_order} (fixed)")
    else:
        print(f"MVAR model order:    Auto-select (AIC)")
    print(f"Skip existing:       {args.skip_existing}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Process each file
    for epoch_file in tqdm(epoch_files, desc="Processing files"):
        pid = epoch_file.stem.replace("_epochs", "")
        
        # Check if output already exists
        if args.skip_existing:
            # Find the relative path to preserve folder structure
            try:
                relative_parent = epoch_file.parent.relative_to(data_dir)
            except ValueError:
                relative_parent = Path()
            
            final_output_dir = output_dir / relative_parent
            metadata_file = final_output_dir / f"{pid}_connectivity_metadata.json"
            
            if metadata_file.exists():
                stats["skipped"] += 1
                continue
        
        stats["processed"] += 1
        file_start = time.time()
        
        # Create output subdirectory (preserve folder structure)
        try:
            relative_parent = epoch_file.parent.relative_to(data_dir)
        except ValueError:
            relative_parent = Path()
        
        final_output_dir = output_dir / relative_parent
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process file
        try:
            success = process_file(
                epoch_file=epoch_file,
                output_dir=final_output_dir,
                model_order=args.model_order
            )
            
            file_time = time.time() - file_start
            stats["processing_times"].append(file_time)
            
            if success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
                stats["failed_files"].append(str(epoch_file))
        
        except Exception as e:
            file_time = time.time() - file_start
            stats["processing_times"].append(file_time)
            stats["failed"] += 1
            stats["failed_files"].append(str(epoch_file))
            tqdm.write(f"✗ ERROR processing {pid}: {e}")
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total files found:    {stats['total_files']}")
    print(f"Files skipped:        {stats['skipped']} (already processed)")
    print(f"Files processed:      {stats['processed']}")
    print(f"Successful:           {stats['successful']}")
    print(f"Failed:               {stats['failed']}")
    
    if stats["processing_times"]:
        avg_time = np.mean(stats["processing_times"])
        print(f"\nTotal time:           {total_time/60:.1f} minutes")
        print(f"Average time/file:    {avg_time:.1f} seconds")
    
    print(f"\nOutput directory:     {output_dir}")
    print("="*70)
    
    # Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_directory": str(data_dir),
        "output_directory": str(output_dir),
        "configuration": {
            "model_order": args.model_order,
            "skip_existing": args.skip_existing
        },
        "statistics": {
            "total_files": stats["total_files"],
            "processed": stats["processed"],
            "skipped": stats["skipped"],
            "successful": stats["successful"],
            "failed": stats["failed"]
        },
        "timing": {
            "total_seconds": total_time,
            "mean_seconds_per_file": float(np.mean(stats["processing_times"])) if stats["processing_times"] else 0
        },
        "failed_files": stats["failed_files"]
    }
    
    summary_path = output_dir / "connectivity_batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}\n")
    
    # Exit with error if all failed
    if stats["successful"] == 0 and stats["processed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
