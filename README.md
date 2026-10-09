# Multilingual Pedagogy Benchmark (branch `multilingual_base`)

This branch is used to **run and maintain the
[Fab AI Multilingual Pedagogy Benchmark](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/benchmarks?benchmarkType=multilingual_pedagogy&benchmark=EN&modelMode=default&price=10&dataset=&models=%5B%5D)**:
the pedagogy benchmark translated into other languages (Arabic, Dari, Hausa, Kiswahili, Luganda,
Nyankore, Pashto, Yoruba, ...), run on many LLMs, and published on the fab-ai.org leaderboard.

## 👉 Start here: [`scripts/README.md`](scripts/README.md)

It explains the whole process step by step, with the commands to run: adding a language,
translating the benchmark (CDPK and SEND), preparing the
dataset, running the models, checking the runs, building the results tables and publishing the
scores to the leaderboard.

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AI-for-Education/pedagogy-benchmark
cd pedagogy-benchmark
git switch multilingual_base
git submodule update --init   # fab-benchmarks-configs: model list, metadata and providers
uv sync                       # installs the versions pinned in uv.lock
uv run dvc pull               # datasets and cached model responses (DVC, Azure remote)
```

`fabdata-llm` (the library that calls the models) comes from its `dev` branch, at the commit pinned
in `uv.lock`, so `uv sync` gives everyone the same version. If you need a newer `fabdata-llm` (e.g. a
new model fails to be called), run `uv sync --upgrade-package fabdata-llm`, test it, and commit
`pyproject.toml` and `uv.lock` together.

### API keys

Copy [`.env.example`](.env.example) to `.env` (same folder) and fill in the keys of the providers you
use; leave the others empty. `.env` is ignored by git, never commit real keys.


## Repository layout

```
├── configs/models/          # lists of models to run (model id: display name)
├── configs/questions/       # one question config per language and category (written by step 3)
├── data/
│   ├── pedagogy_benchmark_full_datasets/  # English source + translated datasets (DVC)
│   ├── <Language>[_ep]/                   # per-category test/dev splits (DVC)
│   ├── translation_log.json               # source version and model of every translation
│   ├── cache/, cache_local/               # cached model responses (shared via DVC / local only)
│   ├── results/                           # benchmark results and multilingual tables
│   └── web/                               # leaderboard JSON files
├── fab-benchmarks-configs/  # submodule: custom_models.yaml, models.csv, providers.csv
├── scripts/                 # pipeline scripts, see scripts/README.md
└── src/cdpk/                # benchmark code (prompts per language, runner, answer parsing)
```

## About the original benchmark

The benchmark comes from the paper
[Benchmarking the Pedagogical Knowledge of Large Language Models](https://arxiv.org/abs/2506.18710):
1,143 questions from teacher qualification exams, split into the **CDPK** benchmark (920 questions,
cross-domain pedagogical knowledge) and the **SEND** benchmark (223 questions, special educational
needs and disabilities). The English questions are on
[Hugging Face](https://huggingface.co/datasets/AI-for-Education/pedagogy-benchmark).

> The documentation of the original English benchmark (single-language runner
> `scripts/run_pedagogy_benchmark.py`, model configuration with fabdata-llm, results layout) is
> **not relevant for this branch** and has been removed from this README. See the `main` branch for
> it. On this branch, use the pipeline described in [`scripts/README.md`](scripts/README.md).


## License

[MIT](LICENSE)
