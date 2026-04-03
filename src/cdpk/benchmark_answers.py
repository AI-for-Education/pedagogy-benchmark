import re
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from fdllm.llmtypes import LLMMessage
from fdllm import get_caller

from cdpk.language_prompts import get_language_config, REPAT

REQ = [re.compile(pat) for pat in REPAT]


def format_questions(df, rowi, question_col, choice_cols, choices):
    prompt = df.iloc[rowi, question_col]
    this_choices_cols = [i for i in choice_cols if not pd.isna(df.iloc[rowi, i])]

    k = len(this_choices_cols)
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[rowi, choice_cols[j]])
    return prompt


def format_example(
    df, rowi, question_col, choice_cols, choices, answer_col, include_answer=True
):
    prompt = format_questions(df, rowi, question_col, choice_cols, choices)
    # prompt += "\nAnswer:"
    if include_answer:
        prompt += "\n{}\n\n".format(df.iloc[rowi, answer_col])
    return prompt


def gen_prompt(
    df, example_rows, question_row, question_col, choice_cols, choices, answer_col,
    language,
):
    # Get language-specific prompts
    config = get_language_config(language)

    # Add custom instructions if English prompt (ep) to explain code-switching
    if "_ep" in language:
        prompt = config['intro_ep'] + "\n\n"
    else:
        prompt = ""

    prompt += config['intro'] + "\n\n"
    for i in example_rows:
        prompt += format_example(df, i, question_col, choice_cols, choices, answer_col)
    prompt += config['instruction'] + "\n\n"
    prompt += format_questions(df, question_row, question_col, choice_cols, choices)
    prompt += "\n\n" + config['final']
    return prompt


def gen_prompt_reasoning_models(df, question_row, question_col, choice_cols, choices):
    prompt = "Answer the following question using the answer format specified below: \n"
    prompt += format_questions(df, question_row, question_col, choice_cols, choices)
    prompt += (
        # "\n\nOnly answer the real question."
        "\n\nOnly provide the letter for your answer."
        "\n\nStop exactly after the letter."
        # "\nDo not provide any explanation."
        # "\nDo not provide any text at all other than the letter by itself."
    )
    return prompt


def clean_resps(resp):
    if pd.isna(resp):
        return
    for req in REQ:
        match = req.match(resp)
        if match is not None:
            break
    if match is None:
        return
    groups = match.groups()
    return groups[0]


def clean_answers(ans):
    return ans.replace(" and", ",").strip()


def evaluate_model(test_df, config, model, verbose=0, language='english'):
    question_col = config["question_col"]
    choice_cols = config["choice_cols"]
    choices = config["choices"]
    answer_col = config["answer_col"]
    example_rows = config["example_rows"]

    caller = get_caller(model)
    total_row = test_df.shape[0]
    # create logical indexer of few-shot rows
    example_filt = np.zeros(len(test_df), dtype=bool)
    example_filt[example_rows] = True
    # drop few-shot examples
    answers = test_df.loc[~example_filt].iloc[:, answer_col]
    resps = list()
    success = list()
    extra_fields = list()
    extra_fields_vars = [
        "Latency",
        "TokensUsed",
        "TokensUsedCompletion",
        "TokensUsedReasoning",
    ]

    for rowi in tqdm(range(len(example_rows), total_row)):

        prompt = gen_prompt(
            test_df,
            example_rows,
            rowi,
            question_col,
            choice_cols,
            choices,
            answer_col,
            language=language,
        )

        try:
            msg = LLMMessage(Role="user", Message=prompt)
            prefixes = ("o1-",
                        "o3-mini",
                        "claude-3-7-sonnet-20250219-thinking-",
                        "o4-mini",
                        "o3-",
                        "claude-sonnet-4-20250514-low",
                        "claude-opus-4-20250514-low",
                        "gpt-5",
                        "claude-sonnet-4-5-20250929-low",
                        "kimi-k2.5",
                        )
            if any(model.startswith(p) for p in prefixes):
                temperature = 1
            else:
                temperature = 0
                
            response = caller.call(msg, max_tokens=None, temperature=temperature)
            
            if verbose > 0:
                print(response.Message)
            resps.append(response.Message)
            success.append(True)
            extra_fields_dict = {
                key: getattr(response, key, None) for key in extra_fields_vars
            }
            extra_fields.append(extra_fields_dict)
        except Exception as e:
            print(e)
            resps.append("")
            success.append(False)
            extra_fields.append({})

        if model == "hunyuan-large-longcontext":
            time.sleep(5)
        if model == "gemini-2.0-pro-exp":
            time.sleep(15)
        if model == "gemini-2.5-pro-exp-03-25":
            time.sleep(15)
        if model == "gemma-3n-e2b-it" or model == "gemma-3n-e4b-it":
            time.sleep(7)

    return answers, resps, extra_fields, success
