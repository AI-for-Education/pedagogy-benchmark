import re
from pathlib import Path
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from fdllm.llmtypes import LLMMessage
from fdllm import get_caller

REPAT = [
    r"^\s*([ABCDEFG])(?:[\.(?:\s*\n)]+.*)*$",
    r"^<think>[\s\S]*?</think>[\s\S]*?([ABCDEFG])$",  # for deepseek R1
    r"^## Step 1[\s\S]*?([ABCDEFG])[\.\s]*$",  # for llama 4
    r'[\s\S]*\n([A-G])"?$',  # for claude 4
]
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
    df, example_rows, question_row, question_col, choice_cols, choices, answer_col
):
    prompt = "The following are example multiple choice questions (with answers).\n\n"
    for i in example_rows:
        prompt += format_example(df, i, question_col, choice_cols, choices, answer_col)
    prompt += "Answer the following real question using same answer format: \n"
    prompt += format_questions(df, question_row, question_col, choice_cols, choices)
    prompt += (
        "\n\nOnly answer the real question."
        "\n\nOnly provide the letter for your answer."
        "\n\nStop exactly after the letter."
        # "\nDo not provide any explanation."
        # "\nDo not provide any text at all other than the letter by itself."
    )
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


def evaluate_model(test_df, config, model, verbose=0):
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

    ## Select specific provider ##
    # DeepSeek
    if model == "or-deepseek-2-5" or model == "deepseek-r1":
        extra_body = {"provider": {"order": ["DeepSeek"], "allow_fallbacks": False}}
    #NovitaAI
    if model == "deepseek-r1-nvt" or model == "minimax-m1" or model == "kimi-k2":
        extra_body = {"provider": {"order": ["NovitaAI"], "allow_fallbacks": False}}
    # DeepInfra
    if model in ["qwen-qwq-32b",
                 "deepseek-v3-0324",
                 "llama-v3p2-1b-instruct-bf16",
                 "llama-v3p2-3b-instruct-bf16",
                 "mistral-nemo-instruct-2407-fp8",
                 "qwen-2p5-7b-instruct-bf16",
                 "mistral-small-24b-instruct-2501-fp8",
                 "qwen3-235b-a22b-thinking-2507",
                 "gemma-3-4b-it",
                 ]:
        extra_body = {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
    if model.startswith("qwen-3-") and model not in ["qwen-3-30b-a3b", "qwen-3-235b-a22b"]:
        extra_body = {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}
    # Parasail
    if model in ["gemma-3-27b",
                 "olmo-2-32b",
                 "qwen-3-235b-a22b-2507-bf16",
                 ]:
        extra_body = {"provider": {"order": ["Parasail"], "allow_fallbacks": False}}
    # Mistral
    if model =="mistral-small-3-1-24b" or model == "mistral-small-3p2-24b-instruct":
        extra_body = {"provider": {"order": ["Mistral"], "allow_fallbacks": False}}
    # xAI
    if model in ["grok-3-beta",
                 "grok-3-mini-beta",
                 ]:
        extra_body = {"provider": {"order": ["xAI"], "allow_fallbacks": False}}
    # Lambda
    if model == "deepseek-r1-0528-fp8-lambda":
        extra_body = {"provider": {"order": ["Lambda"], "allow_fallbacks": False}}
    else:
        extra_body = {}

    for rowi in tqdm(range(len(example_rows), total_row)):

        prompt = gen_prompt(
            test_df,
            example_rows,
            rowi,
            question_col,
            choice_cols,
            choices,
            answer_col,
        )
        try:
            msg = LLMMessage(Role="user", Message=prompt)
            if (
                model.startswith("o1-")
                or model.startswith("o3-mini")
                or model.startswith("claude-3-7-sonnet-20250219-thinking-")
                or model.startswith("o4-mini")
                or model.startswith("o3-")
                or model.startswith("gpt-5-")
            ):
                temperature = 1
            else:
                temperature = 0
            if extra_body:
                response = caller.call(
                    msg, max_tokens=None, temperature=temperature, extra_body=extra_body
                )
            else:
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

    return answers, resps, extra_fields, success
