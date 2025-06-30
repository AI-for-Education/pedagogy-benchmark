from collections import defaultdict
from pathlib import Path, PureWindowsPath

import pandas as pd
import yaml
import numpy as np

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
        answers, resps, success = evaluate_model(df, config=config, model=model, verbose=0)
        try:
            df_res = pd.DataFrame(
                {"answers": answers, "resps": resps, "success": success}
            ).reset_index(drop=True)
            if use_cache:
                outfile.parent.mkdir(exist_ok=True, parents=True)
                df_res.to_csv(outfile, index=False)
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
        pred_sr = pd.Series(index=df.index, name=f"pred_{model}", dtype=object)
        example_filt = np.zeros(len(df), dtype=bool)
        example_filt[config["example_rows"]] = True
        pred_sr[example_filt] = "Few-shot example"
        pred_sr[~example_filt] = df_res.loc[:, "resps"].to_numpy()
        predlist.append(pred_sr)

    df = pd.concat([df, *predlist], axis=1)
    summary_df = pd.DataFrame(resdict)
    full_df = df

    return full_df, summary_df, config, models
