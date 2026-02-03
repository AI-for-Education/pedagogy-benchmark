import json
from pathlib import Path
from collections import defaultdict

# Set up paths
DATA_DIR = Path("data")
output_file = DATA_DIR / "model_latencies_merged.json"

# Find all language-specific latency files
latency_files = list(DATA_DIR.glob("model_latencies_*.json"))
# Exclude the merged file if it already exists
latency_files = [f for f in latency_files if f.name != "model_latencies_merged.json"]

print(f"Found {len(latency_files)} latency files to merge:")
for f in latency_files:
    print(f"  - {f.name}")

# Initialize merged dictionary
merged_latencies = {}

# Merge all files
for file_path in latency_files:
    print(f"\nProcessing {file_path.name}...")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Merge into the main dictionary
        for model_name, configs in data.items():
            if model_name not in merged_latencies:
                merged_latencies[model_name] = {}

            # Merge the configs for this model
            for config_name, latency in configs.items():
                if config_name in merged_latencies[model_name]:
                    print(f"  Warning: Duplicate entry for {model_name}/{config_name}")
                    print(f"    Existing: {merged_latencies[model_name][config_name]:.2f}s")
                    print(f"    New: {latency:.2f}s")
                    print(f"    Keeping existing value")
                else:
                    merged_latencies[model_name][config_name] = latency

        print(f"  Added {len(data)} models from {file_path.name}")

    except json.JSONDecodeError as e:
        print(f"  Error reading {file_path.name}: {e}")
    except Exception as e:
        print(f"  Error processing {file_path.name}: {e}")

# Save merged file
print(f"\nSaving merged latencies to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(merged_latencies, f, indent=4)

print(f"✓ Successfully merged {len(merged_latencies)} models")
print(f"  Total configurations: {sum(len(configs) for configs in merged_latencies.values())}")
