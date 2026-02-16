#!/usr/bin/env python3
"""
Script to merge CSV files from data/cache_local to data/cache.

Usage:
    uv run scripts/merge_cache.py

Description:
    For each folder in cache_local:
    - If the folder doesn't exist in cache: create it and copy all CSV files
    - If the folder exists in cache: only copy CSV files that aren't already present
"""

import os
import shutil
from pathlib import Path


def merge_cache():
    """Merge CSV files from cache_local to cache."""
    # Define paths
    cache_local_dir = Path("data/cache_local")
    cache_dir = Path("data/cache")

    # Check if cache_local exists
    if not cache_local_dir.exists():
        print(f"Error: {cache_local_dir} does not exist")
        return

    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get all folders in cache_local
    folders = [f for f in cache_local_dir.iterdir() if f.is_dir()]

    print(f"Found {len(folders)} folders in {cache_local_dir}")

    total_copied = 0
    total_skipped = 0

    for source_folder in folders:
        folder_name = source_folder.name
        target_folder = cache_dir / folder_name

        # Get all CSV files in the source folder
        csv_files = list(source_folder.glob("*.csv"))

        if not csv_files:
            print(f"  {folder_name}: No CSV files found, skipping")
            continue

        # Check if target folder exists
        if not target_folder.exists():
            # Create folder and copy all CSV files
            print(f"  {folder_name}: Creating folder and copying {len(csv_files)} CSV files")
            target_folder.mkdir(parents=True, exist_ok=True)

            for csv_file in csv_files:
                target_file = target_folder / csv_file.name
                shutil.copy2(csv_file, target_file)
                total_copied += 1
                print(f"    ✓ Copied {csv_file.name}")
        else:
            # Only copy files that don't exist in target
            copied_count = 0
            skipped_count = 0

            for csv_file in csv_files:
                target_file = target_folder / csv_file.name

                if not target_file.exists():
                    shutil.copy2(csv_file, target_file)
                    copied_count += 1
                    total_copied += 1
                    print(f"    ✓ Copied {csv_file.name}")
                else:
                    skipped_count += 1
                    total_skipped += 1

            if copied_count > 0 or skipped_count > 0:
                print(f"  {folder_name}: Copied {copied_count} files, skipped {skipped_count} existing files")

    print(f"\nSummary:")
    print(f"  Total CSV files copied: {total_copied}")
    print(f"  Total CSV files skipped (already exist): {total_skipped}")


if __name__ == "__main__":
    merge_cache()
