# Scripts Directory

This directory contains scripts for preparing datasets, running benchmarks, and analyzing results for the multilingual pedagogy benchmark.

## 📌 Important Note

**Most scripts are designed to be run in an interactive environment (Jupyter notebook, IPython, or VS Code interactive window).**

Only `run_pedagogy_benchmark_multilingual.py` should be run from the command line.

---

## Dataset Preparation

### `prepare_cdpk_dataset_multilingual.py`
🔧 **Run in interactive window** (Jupyter/IPython/VS Code)

Prepares CDPK (Child Development and Pedagogical Knowledge) dataset for multilingual benchmarking.

**What it does:**
1. Loads the pedagogy benchmark CSV for the specified language
2. Splits data by category (Science, Literacy, Creative arts, Maths, Social studies, Technology, General)
3. Creates dev/test splits for each category
4. Generates YAML configuration files in `configs/questions/`

**Usage:**
Open the file in an interactive environment and modify the language configuration at the top of the script, then run the cells.

---

## Running Benchmarks

### `run_pedagogy_benchmark_multilingual.py`
⚡ **Run from command line**

Runs the pedagogy benchmark evaluation for a specific language across multiple models.

**Usage:**
```bash
python run_pedagogy_benchmark_multilingual.py --language <language_code> [options]
```

**Key Options:**
- `--language`: Target language (required)
- `--benchmark`: Type of benchmark (`cdpk` or `send`, default: `cdpk`)
- `--categories`: Categories to evaluate (default: all 7 categories)
- `--models-config`: Models configuration file name (optional)
- `--max-workers`: Number of parallel workers (default: 20)
- `--save-results`: Save detailed results to disk

**What it does:**
1. Loads language-specific prompts and configurations
2. Runs benchmark evaluation on specified models
3. Calculates accuracy by category
4. Saves results to `data/results/`

**Example:**
```bash
# Run benchmark for English on all categories
python run_pedagogy_benchmark_multilingual.py --language english --save-results

# Run benchmark for Swahili on science category only
python run_pedagogy_benchmark_multilingual.py --language swahili_ep --categories science
```

### `run_pedagogy_benchmark.py`
⚠️ **LEGACY - Not recommended for use**

This is a legacy script with hardcoded configurations. **Use `run_pedagogy_benchmark_multilingual.py` instead** for all benchmark runs.

---

## Analysis & Results

### `create_results.py`
🔧 **Run in interactive window** (Jupyter/IPython/VS Code)

Generates analysis results, figures, and aggregated metrics from benchmark runs.

**What it does:**
- Loads cached results from `data/cache_local/`
- Computes accuracy metrics by category, language, and model
- Creates visualizations and summary tables
- Exports results to `data/results/`

### `create_figures.py`
🔧 **Run in interactive window** (Jupyter/IPython/VS Code)

Creates publication-quality figures and plots from benchmark results.

**What it does:**
- Generates bar charts, heatmaps, and comparison plots
- Visualizes model performance across languages and categories
- Saves figures to `data/results/`

---

## Data Loading & Exploration

### `load_dataset.py`
🔧 **Run in interactive window** (Jupyter/IPython/VS Code)

Utility script for loading and exploring the pedagogy benchmark dataset.

**What it does:**
- Demonstrates how to load dataset from CSV files
- Shows data structure and available fields
- Useful for interactive exploration

### `load_dataset_multingual_human_reviews_exp.py`
🔧 **Run in interactive window** (Jupyter/IPython/VS Code)

Specialized script for loading and exploring human-reviewed multilingual datasets.

**What it does:**
- Loads reviewed datasets with quality scores
- Analyzes human review annotations
- Useful for quality assurance

---

## Analysis Scripts

### `analysis_human_vs_machine_translations.py`
Compares model performance on human-translated vs machine-translated datasets.

**What it does:**
- Evaluates translation quality impact
- Compares accuracy across translation methods
- Generates comparative analysis

### `baseline_allmodels_general.py`
Runs baseline evaluation across all models on general questions.

**What it does:**
- Evaluates all configured models
- Focuses on general category questions
- Produces baseline metrics

---

## Utilities

### `check_model_config.py`
Validates model configuration files.

**What it does:**
- Checks if model configs are properly formatted
- Verifies model availability
- Lists available models

**Usage:**
```bash
python check_model_config.py
```

### `translate_benchmark.py`
🔧 **Run in interactive window** (Jupyter/IPython/VS Code)

Translates benchmark questions to other languages.

**What it does:**
- Translates questions using specified translation service
- Maintains question structure and format
- Saves translated datasets

---

## Workflow Overview

### Complete Pipeline

1. **Prepare Dataset** (Interactive)
   - Open `prepare_cdpk_dataset_multilingual.py` in interactive window
   - Configure language at the top of the script
   - Run cells to generate dataset

2. **Run Benchmark** (Command Line)
   ```bash
   python run_pedagogy_benchmark_multilingual.py --language english --save-results
   ```

3. **Generate Results** (Interactive)
   - Open `create_results.py` in interactive window
   - Run cells to generate analysis
   - Open `create_figures.py` in interactive window
   - Run cells to create visualizations

### Supported Languages

- **English**: `english`
- **Luganda**: `luganda`, `luganda_ep` (English prompt)
- **Swahili**: `swahili`, `swahili_ep` (English prompt)
- **Hausa**: `hausa`, `hausa_ep` (English prompt)
- **Yoruba**: `yoruba`, `yoruba_ep` (English prompt)
- **Nyankore**: `nyankore`, `nyankore_ep` (English prompt)

**Note:** `_ep` suffix indicates "English Prompt" version where instructions are in English but questions remain in the target language.

---

## Output Locations

- **Prepared datasets**: `data/<language>/CDPK_per_category/`
- **YAML configs**: `configs/questions/`
- **Cached results**: `data/cache_local/`
- **Final results**: `data/results/`
- **Figures**: `data/results/`

---

## Requirements

All scripts require dependencies from the project root:
```bash
pip install -r requirements.txt
```

Make sure to set up your `.env` file with necessary API keys before running benchmarks.
