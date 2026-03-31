"""
Count rows with success == False in cache CSV files.

Usage:
    # Scan all folders in data/cache_local/
    python scripts/count_failures.py

    # Scan only folders matching specific prefixes (matches all subfolders starting with the prefix)
    python scripts/count_failures.py CDPK_Hausa_ep CDPK_Swahili_ep CDPK_Yoruba_ep CDPK_Nyankore_ep CDPK_Luganda_ep CDPK_English

    # Scan specific single folder
    python scripts/count_failures.py CDPK_English

    # Use a different cache directory
    python scripts/count_failures.py --cache-dir data/other_cache CDPK_Hausa_ep

Output:
    Table of CSV files that have at least one False in the 'success' column,
    sorted from fewest to most failures.
"""

import os
import csv
import argparse


def count_false_rows(cache_dir: str, folder_prefixes: list[str] | None = None) -> list[tuple[str, int]]:
    results = []

    if folder_prefixes:
        # collect folders whose name starts with any of the given prefixes
        dirs_to_scan = []
        for entry in sorted(os.listdir(cache_dir)):
            full = os.path.join(cache_dir, entry)
            if os.path.isdir(full) and any(entry.startswith(p) for p in folder_prefixes):
                dirs_to_scan.append(full)
    else:
        dirs_to_scan = [cache_dir]

    for scan_dir in dirs_to_scan:
        for root, _, files in os.walk(scan_dir):
            for fname in files:
                if not fname.endswith(".csv"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        if "success" not in (reader.fieldnames or []):
                            continue
                        count = sum(1 for row in reader if row["success"] == "False")
                except Exception as e:
                    print(f"WARNING: could not read {fpath}: {e}")
                    continue
                if count > 0:
                    results.append((fpath, count))

    results.sort(key=lambda x: x[1])
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Count False success rows in cache CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="Folder name prefixes to scan (e.g. CDPK_Hausa_ep CDPK_Swahili_ep). "
             "If omitted, scans everything in cache_dir.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "cache_local"),
        help="Path to cache directory (default: data/cache_local)",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    prefixes = args.folders if args.folders else None

    results = count_false_rows(cache_dir, prefixes)

    if not results:
        print("No files with False success rows found.")
        return

    # shorten paths relative to cache_dir for readability
    display = [(os.path.relpath(p, cache_dir), c) for p, c in results]

    max_path = max(len(d[0]) for d in display)
    header_path = "CSV File"
    header_count = "False Count"
    col_w = max(max_path, len(header_path))

    print(f"{header_path:<{col_w}}  {header_count}")
    print(f"{'-' * col_w}  {'-' * len(header_count)}")
    for path, count in display:
        print(f"{path:<{col_w}}  {count}")
    print(f"\nTotal files with failures: {len(results)}")


if __name__ == "__main__":
    main()
