from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CACHE_LOCAL_DIR = ROOT / "data" / "cache_local"
CACHE_GLOBAL_DIR = ROOT / "data" / "cache"

provider_df = pd.read_csv(ROOT / "data" / "providers.csv")
models_df = pd.read_csv(ROOT / "data" / "models.csv")

PROVIDER_MODEL_MAPPING = {}
for _, row in models_df.iterrows():
    provider = row["provider"]
    model = row["model_id"]
    if provider not in PROVIDER_MODEL_MAPPING:
        PROVIDER_MODEL_MAPPING[provider] = []
    PROVIDER_MODEL_MAPPING[provider].append(model)

PROVIDER_COLOR_MAPPING = {
    row["provider"]: row["color"] for _, row in provider_df.iterrows()
}

PROVIDER_SVG_MAPPING = {
    row["provider"]: row["svg"] for _, row in provider_df.iterrows()
}


MODEL_COST_MAPPING = {}
for _, row in models_df.iterrows():
    model = row["model_id"]
    input_cost = row["input_cost"]
    output_cost = row["output_cost"]
    if pd.notnull(output_cost):
        MODEL_COST_MAPPING[model] = [input_cost, output_cost]
    elif pd.notnull(input_cost):
        MODEL_COST_MAPPING[model] = [input_cost]
    else:
        MODEL_COST_MAPPING[model] = None

MODEL_SIZE_MAPPING = {
    row["model_id"]: row["size"] for _, row in models_df.iterrows() if not pd.isna(row["size"])
}

MODEL_OPEN_MAPPING = {row["model_id"]: row["open"] for _, row in models_df.iterrows()}

MODEL_URL_MAPPING = {row["model_id"]: row["url"] for _, row in models_df.iterrows()}

MODEL_RELEASE_DATA_MAPPING = {
    row["model_id"]: row["release_date"]
    for _, row in models_df.iterrows()
    if row["release_date"]
}
