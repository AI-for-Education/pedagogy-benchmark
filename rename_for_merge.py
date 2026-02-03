#!/usr/bin/env python3
"""
Simple script to rename conflicting files with current branch suffix
Usage:
    git checkout <branch>
    python rename_for_merge.py
"""

import subprocess
import os
from pathlib import Path

# Get current branch name
result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                       capture_output=True, text=True, check=True)
branch = result.stdout.strip()

print(f"Current branch: {branch}")
print("Renaming conflicting files...")
print()

# Files to rename
files_to_rename = [
    ".gitignore",
    "configs/models/full_list_20251015.yaml",
    "configs/models/full_list_20251015_small.yaml",
    "custom_models.yaml",
    "data/model_latencies.json",
    "scripts/prepare_cdpk_dataset_multilingual.py",
    "scripts/run_pedagogy_benchmark_multilingual.py",
    "scripts/translate_benchmark.py",
    "src/cdpk/benchmark_answers.py",
    "src/cdpk/benchmark_run.py",
]

renamed_count = 0

for file_path in files_to_rename:
    if os.path.exists(file_path):
        # Get directory and filename
        path = Path(file_path)
        directory = path.parent
        name = path.stem
        extension = path.suffix

        # Create new filename with branch suffix
        new_filename = f"{name}_{branch}{extension}"
        new_path = directory / new_filename

        # Rename using git mv
        subprocess.run(['git', 'mv', file_path, str(new_path)], check=True)
        print(f"[OK] {path.name}")
        renamed_count += 1

print()
if renamed_count > 0:
    print("Committing changes...")
    subprocess.run(['git', 'commit', '-m',
                   f'Rename conflicting files with _{branch} suffix for merge'],
                   check=True)
    print()
    print(f"✓ Done! {renamed_count} files renamed on branch: {branch}")
else:
    print("No files to rename.")
