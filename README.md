
# Pedagogy Benchmarks

This repository contains code for the paper [Benchmarking the Pedagogical Knowledge of Large Language Models](https://arxiv.org/abs/2506.18710). 

There is a script to run both the Cross-Domain Pedagogical Knowledge (CDPK) benchmark and the Special Educational Needs and Disabilities (SEND) benchmark. It also contains the code used to extract and preprocess the questions, as described in the paper. 

Current results are shown on the leaderboards: 
- [CDPK](https://rebrand.ly/pedagogy)
- [SEND](https://rebrand.ly/sendpedagogy)

A simplified standalone dataset containing the questions is available at [HuggingFace](https://huggingface.co/datasets/AI-for-Education/pedagogy-benchmark).

We request that you *do not reveal examples from this dataset online*, to reduce the risk of leakage into foundation model training corpora. 

## Overview

This benchmark uses questions testing pedagogical knowledge from teacher qualification exams. The dataset contains 1143 questions spanning a range of subjects (or categories), age groups (e.g. primary or secondary school level) and pedagogical subdomains. As explained in the paper, we split these questions into two benchmarks. 

- **CDPK Benchmark** (920 questions): Tests general pedagogical knowledge across multiple subject areas.
- **SEND Benchmark** (223): Questions from the Special Educational Needs and Disabilities category.

The system supports 100+ models from various providers including OpenAI, Anthropic, Google, Meta, and many others, using the [fabdata-llm](https://github.com/AI-for-Education/fabdata-llm) package.

This repository uses [DVC](https://dvc.org/) for data version control, to provide up-to-date read-only access to the raw output and results from all the models we have run the benchmark on ourselves. 

## Quick Start

### Prerequisites

- Python ≥3.13
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AI-for-Education/pedagogy-benchmark
cd pedagogy-benchmark
```

2. Install dependencies:
```bash
uv sync
```

3. Pull existing results:
```bash
uv run dvc pull
```

4. Set up your environment variables in `.env`:
```bash
# API keys for various providers
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
# ... other provider keys (see fabdata-llm and custom_models.yaml)


```

### Running the Benchmarks

#### Model Configs

A benchmark can be run on a set of models, which are defined in a yaml file in the `configs/models` folder. This contains a dictionary where the keys are the `fabdata-llm` model name (i.e. see `custom_models.yaml`), and the values are a nicely formatted model display name.

#### Run Benchmark Script

The main entry point for running benchmarks is `scripts/run_pedagogy_benchmark.py`. 

```bash
# Run CDPK benchmark with default models
uv run python scripts/run_pedagogy_benchmark.py --benchmark cdpk

# Run SEND benchmark
uv run python scripts/run_pedagogy_benchmark.py --benchmark send --models-config send_online_leaderboard

# Use a specific models configuration
uv run python scripts/run_pedagogy_benchmark.py --benchmark cdpk --models-config cdpk_online_leaderboard

# Specify custom output folder
uv run python scripts/run_pedagogy_benchmark.py --benchmark cdpk --output-folder ./my_results
```

For each model in the specified `models-config` file, this script will look for existing results in `data/cache_local` or `data/cache` as `<questions_config>/results_<model_id>.csv`. If results are found, the script parses the outputs to extract the answer for each question and computes the accuracy of the model on the benchmark. If no results are found, the benchmark will be run for that model. Each question is evaluated with an independent API call to the model.  

If a new model is run from this repository, the results are stored in `data/cache_local`. This is because `data/cache` is tracked with dvc with a read-only remote, so local changes would cause a conflict if you later try to update with new models that we run as we continue to update the online leaderboard. Storing model outputs in `cache_local` avoids this. In case of duplicate model runs, your local runs in `cache_local` take priority over our central runs in `cache`.

The results are written in `data/results/<benchmark>_results_{accuracy,bad_format,full}_<model_config>.csv`. `accuracy` gives the per-subject and overall accuracy. `bad_format` gives the per-subject and overall percentage of questions which were not parsed successfully. `full` gives the full question-level results (e.g. to allow correlating errors between different models).

## Repository Structure 

```
├── configs/            # Configuration files for models and questions
├── data/               # Benchmark data, results, and cached responses
├── scripts/            # Main scripts for running benchmarks and analysis
├── scripts/legacy      # Code used for development of the benchmark
├── src/                # Project module
├── custom_models.yaml  # Configuration for custom LLMs
├── pyproject.toml      # Project dependencies and metadata
└── README.md           # Detailed project documentation
```


### Key Directories and Files:

*   **`configs/`**: Contains YAML files that define the models to be tested (`configs/models`) and the questions to be used in the benchmarks (`configs/questions`).
*   **`data/`**: 
    *   `data/cache/`: DVC-tracked read-only cache of model responses.
    *   `data/cache_local/`: non-DVC tracked local cache.
    *   `data/results/`: CSV files with the benchmark results.
*   **`scripts/run_pedagogy_benchmark.py`**: The main entry point for running the CDPK and SEND benchmarks.
*   **`src/cdpk/`**: The core Python package containing the logic for running the benchmarks, processing answers, and evaluating models.
    *   `src/cdpk/benchmark_run.py`: Handles the execution of a benchmark for a given set of models and questions.
    *   `src/cdpk/benchmark_answers.py`: Contains functions for cleaning and evaluating model responses.
*   **`custom_models.yaml`**: Defines the configuration for LLMs, including API endpoints, keys, and other parameters. This file is used by the `fabdata-llm` library.
*   **`pyproject.toml`**: Defines the project's Python dependencies. Use `uv sync` to install them.

## Model Configuration

This benchmark uses the [Fabdata-LLM](https://github.com/AI-for-Education/fabdata-llm) library for model interactions. Fadata-LLM supports 100+ models across a range of providers, including any OpenAI API compatible endpoint.

A range of models are pre-defined in the fabdata-llm [config](https://github.com/AI-for-Education/fabdata-llm/blob/main/src/fdllm/models.yaml), and available by default, but this might not include the newest models. Additional models can be added to `custom_models.yaml`. This file includes the configuration for all the models run so far on the benchmark. 

### Example model configuration

As an example, any endpoint using the OpenAI API can be configured as follows:

```yaml
MyModelSet:
  Api_Interface: OpenAI
  Api_Key_Env_Var: <provider_api_key_env_var_name>
  Client_Args:
    base_url: <OpenAI compatible base_url>
  models:
    <model_id>:
      Token_Window: <context window length>
      Token_Limit_Completion: <max completion tokens>
      Api_Model_Name: <api_model_name> # if different from model_id
```

The name of the model set is arbitrary, as the grouping just provides a way to share config options across a list of models. Any model config option can be set for an individual model, or at the model-set level. Options include `Client_Args` (a dict that is parsed through at API client initialisation), and `Call_Args` which are passed to the API call function. For example, if using OpenRouter, a specific provider should be selected as follows (to avoid different questions being routed to different providers which might be running different model quantisations):

```yaml
OpenRouter:
  Api_Interface: OpenAI
  Api_Key_Env_Var: OPENROUTER_API_KEY
  Client_Args:
    base_url: https://openrouter.ai/api/v1
  models:
    deepseek-r1:
      Token_Window: 131000
      Token_Limit_Completion: 131000
      Api_Model_Name: deepseek/deepseek-r1
      Call_Args:
        extra_body:
          provider:
            order:
              - Deepseek
            allow_fallbacks: False
```

#### Local models

Fabdata-LLM doesn't have explicit support for local models, but if your model runner provides an OpenAI compatible endpoint on localhost, you could configure it as above. 

### Steps to run a new model

To run a new model:

1. Add model configuration to `custom_models.yaml`
2. Test model access with `scripts/check_model.py`
```bash
uv run python scripts/check_model.py --model <model_id>
```
3. Add model to an existing model config yaml file in `configs/models`, or create one for a single model:
```bash
echo "<model_id>:<model_display_name>" > configs/models/mymodel.yaml
```
4. Run the benchmark for that model config
```bash
uv run python scripts/run_pedagogy_benchmark.py --benchmark cdpk --models-config mymodel
```

## Results and Outputs

When you run a benchmark, results are saved to:
- **Raw model outputs**: `data/cache_local/<question_config>/resps_<model_id>.csv` with columns:
  - `answer`: ground truth (correct answer)
  - `resps`: raw model response
  - `success`: indicates whether the call to the model completed successfully. 
- **Accuracy**: `data/results/<benchmark>_results_accuracy_<model_list>.csv` with columns for accuracy of each category, and overall accuracy. 
- **Percentage bad format questions**: `data/results/<benchmark>_bad_format_<model_list>.csv` with columns for percentage of un-parseable answers in each category, and overall. 
- **Full question-level results**: `data/results/<benchmark>_full_<model_list>.csv` with a row for each question and columns with each model's raw response, as well as the metadata associated with that question. 

You can run `git pull` and `uv run dvc pull` to update with the latest models we have tested. 

## Code used for benchmark development


The `scripts/legacy/` directory contains code used for preprocessing the questions during the development of the benchmark:

- **Question Generation**: Processing and converting educational materials into benchmark format
- **Data Preprocessing**: Cleaning and preparing datasets
- **Duplicate Detection**: Identifying and handling duplicate questions
- **Leaderboard Maintenance**: Analyzing results and updating performance metrics
- **Quality Assurance**: Validation and verification of benchmark data

## Citation

If you find this useful in your research, please consider citing the paper:

```
 @misc{lelievre2025pedagogybenchmark,
      title={Benchmarking the Pedagogical Knowledge of Large Language Models}, 
      author={Maxime Lelièvre and Amy Waldock and Meng Liu and Natalia Valdés Aspillaga and Alasdair Mackintosh and María José Ogando Portela and Jared Lee and Paul Atherton and Robin A. A. Ince and Oliver G. B. Garrod},
      year={2025},
      eprint={2506.18710},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.18710}, 
}
```

## Acknowledgements

We thank the [Agencia de la Calidad de Educación y Centro de Perfeccionamiento, Experimentación e Investigaciones Pedagógicas (Education Quality Agency and Center for Pedagogical Improvement, Experimentation, and Research)](https://www.cpeip.cl/cpeip/), of the Chilean Ministry of Education for developing and sharing the exams this benchmark is based on. 

## Contact

Any questions or comments please contact: [benchmarks@ai-for-education.org](mailto:benchmarks@ai-for-education.org)

## License

[MIT](LICENSE)

---

**Note**: This benchmark is designed for research and evaluation purposes. Ensure you have appropriate API access and understand the costs associated with running large-scale model evaluations.