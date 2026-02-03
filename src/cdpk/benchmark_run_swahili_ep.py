from collections import defaultdict
from pathlib import Path, PureWindowsPath
from xml.parsers.expat import model

import pandas as pd
import yaml
import numpy as np
import time
import json

from .benchmark_answers import clean_resps, clean_answers, evaluate_model
from .benchmark_constants import ROOT, CACHE_GLOBAL_DIR, CACHE_LOCAL_DIR


def process_path(path_str):
    if path_str is None:
        return
    path = Path(PureWindowsPath(path_str).as_posix())
    if not path.exists():
        path = ROOT / "data" / path.as_posix()
        #path.relative_to(ROOT / "data")
        if not path.exists():
            raise ValueError(f"Path {str(path)} does not exist")
    return path


def load_from_config(
    questions_config="MMLU_abstract_algebra", models_config="fast_cheap"
):
    questions_config = ROOT / "configs/questions" / f"{questions_config}.yaml"
    with open(questions_config) as f:
        config = yaml.safe_load(f)

    models_config = ROOT / "configs/models" / f"{models_config}.yaml"
    with open(models_config) as f:
        models = yaml.safe_load(f)

    test_file = process_path(config["test_file"])
    example_file = process_path(config["example_file"])

    df = pd.read_csv(test_file, header=config["test_header"])
    if example_file is not None:
        example_df = pd.read_csv(example_file, header=config["example_header"])
        if config["example_rows"]:
            df = pd.concat(
                [example_df.iloc[config["example_rows"]], df], axis=0, ignore_index=True
            )

    return df, config, models


def run_benchmark(
    questions_config="CDPK_gen_pk",
    models_config="full_list_20250106",
    verbose=1,
    use_cache=True,
):
    def runner():
        # Record the start time
        start_time = time.time()

<<<<<<<< HEAD:src/cdpk/benchmark_run_luganda.py
<<<<<<<< HEAD:src/cdpk/benchmark_run_luganda.py
<<<<<<<< HEAD:src/cdpk/benchmark_run_multilingual_base.py
        answers, resps, success = evaluate_model(df, config=config, model=model, verbose=0)
========
        answers, resps, extra_fields, success = evaluate_model(df, config=config, model=model, verbose=0)
>>>>>>>> luganda:src/cdpk/benchmark_run_luganda.py
========
        answers, resps, extra_fields, success = evaluate_model(df, config=config, model=model, verbose=0)
>>>>>>>> luganda_ep:src/cdpk/benchmark_run_luganda_ep.py
========
        answers, resps, extra_fields, success = evaluate_model(df, config=config, model=model, verbose=0)
>>>>>>>> swahili_ep:src/cdpk/benchmark_run_swahili_ep.py
        try:
            df_res_basic = pd.DataFrame(
                {"answers": answers, "resps": resps, "success": success}
            ).reset_index(drop=True)
            df_res_extra = pd.DataFrame(extra_fields).reset_index(drop=True)
            df_res = pd.concat([df_res_basic, df_res_extra], axis=1)
            if use_cache:
                outfile.parent.mkdir(exist_ok=True, parents=True)
                df_res.to_csv(outfile, index=False)

            # ---- Start of modified code for timing ----

            # 1. Record the end time and calculate the total latency in seconds
            end_time = time.time()
            latency = end_time - start_time

            # 2. Define the path to your JSON file
            latency_file_path = Path("data/model_latencies.json")

            # 3. Ensure the 'data' directory exists
            latency_file_path.parent.mkdir(exist_ok=True, parents=True)

            # 4. Safely load existing data or create a new dictionary
            try:
                # Check if file exists and is not empty before trying to load
                if latency_file_path.exists():
                    with open(latency_file_path, 'r') as f:
                        latencies = json.load(f)
                else:
                    latencies = {}
            except json.JSONDecodeError:
                # If file is corrupted or malformed, start fresh
                latencies = {}

            # 5. Define the name for your question configuration here.
            #    This variable should hold the name of the MCQ set being evaluated.
            questions_config_name = questions_config

            # 6. Add the new nested latency value
            #    First, ensure the model key exists (initialize it as a dict if not)
            if model not in latencies.keys():
                latencies[model] = {}

            #    Then, add the latency for the specific question config
            latencies[model][questions_config_name] = latency

            # 7. Write the updated dictionary back to the JSON file
            with open(latency_file_path, 'w') as f:
                json.dump(latencies, f, indent=4)

            # ---- End of modified code ----

            return df_res
        except:
            return

    df, config, models = load_from_config(questions_config, models_config)

    resdict = defaultdict(dict)
    predlist =[]

    for model in models:
        if verbose > 0:
            print(model)
        outfile = CACHE_LOCAL_DIR / questions_config / f"resps_{model}.csv"

        ### check global cache before local cache, local cache overrides global cache
        cache_check_files = [
            CACHE_GLOBAL_DIR / outfile.relative_to(CACHE_LOCAL_DIR), outfile
        ]

        if use_cache:
            cache_hit = False
            for check_file in cache_check_files: 
                if check_file.exists():
                    df_res = pd.read_csv(check_file)
                    cache_hit = True
            if not cache_hit:
                df_res = runner()
                if df_res is None:
                    continue
        else:
            df_res = runner()
            if df_res is None:
                continue

        df_res["resps"] = df_res["resps"].apply(clean_resps)
        df_res["answers"] = df_res["answers"].apply(clean_answers)

        resdict[model]["accuracy"] = (df_res["resps"] == df_res["answers"]).mean()
        resdict[model]["bad_format"] = df_res["resps"].isna().mean()
        if verbose > 0:
            print(f'Accuracy: {resdict[model]["accuracy"]:.3f}')
            print(
                f'Badly formatted: {resdict[model]["bad_format"]:.03f}'
            )
            print()

        #### create series to hold the model answers (pred), accounting for the
        #### few-shot examples indices
        example_filt = np.zeros(len(df), dtype=bool)
        example_filt[config["example_rows"]] = True
        pred_sr = pd.Series(index=df.index, name=f"pred_{model}", dtype=object)
        pred_sr[example_filt] = "Few-shot example"
        pred_sr[~example_filt] = df_res.loc[:, "resps"].to_numpy()
        predlist.append(pred_sr)
        for var in df_res.columns:
            if var not in ["resps", "success", "answers"]:
                var_sr = pd.Series(index=df.index, name=f"{var}_{model}", dtype=object)
                var_sr[example_filt] = "Few-shot example"
                var_sr[~example_filt] = df_res.loc[:, var].to_numpy()
                predlist.append(var_sr)

    df = pd.concat([df, *predlist], axis=1)
    summary_df = pd.DataFrame(resdict)
    full_df = df

    return full_df, summary_df, config, models
