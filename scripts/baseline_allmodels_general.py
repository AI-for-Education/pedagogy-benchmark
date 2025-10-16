# %%
from pathlib import Path

from dotenv import load_dotenv
from fdllm.sysutils import register_models, list_models

## fixing path issue 
import sys
# Add the parent directory's 'src' folder to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
####
from cdpk.benchmark_run import run_benchmark

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
dotenv_path = ROOT / ".env"
load_dotenv(dotenv_path, override=True)

project_models = ROOT / "custom_models.yaml"
home_models = Path.home() / ".fdllm" / "custom_models.yaml"
# load a project custom_models if it exists
# otherwise fall back to home directory (on maximes computer)
if project_models.exists():
    register_models(project_models)
elif home_models.exists():
    register_models(home_models)

    
# %%
language = "Luganda"
# run CDPK benchmarks
CDPK_list = [
    f"CDPK_{language}_science",
    f"CDPK_{language}_literacy",
    f"CDPK_{language}_creative_arts",
    f"CDPK_{language}_maths",
    f"CDPK_{language}_social_studies",
    f"CDPK_{language}_technology",
    f"CDPK_{language}_gen_pk",
]

for config in CDPK_list:
    print(f"Running CDPK benchmark for {config}")
    full_df, summary_df, config, models = run_benchmark(
        questions_config=config, models_config="full_list_20251015"
    )
    break


# %%
# running SEND benchmark
config = "CDPK_send"
full_df, summary_df, config, models = run_benchmark(
        questions_config=config, models_config="full_list_20250627_send"
    )

# %%
from fdllm import get_caller
from fdllm.llmtypes import LLMMessage

model = "mistral-medium-3.1"
caller = get_caller(model)
prompt = "Can you tell me a joke about cooks and light bulbs?"
response = caller.call(LLMMessage(Role="user", Message=prompt))
print(response.Message)

