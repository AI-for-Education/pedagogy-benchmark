#
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv
from fdllm.sysutils import register_models
from cdpk.benchmark_constants import ROOT
from fdllm import get_caller, LLMMessage

load_dotenv(override=True)

TEST_PROMPT = "Tell me a joke about cooks and lightbulbs."

def main(opt):
    model = opt.model
    if opt.custom_models_file:
        custom_models_file = Path(opt.custom_models_file)
        if not custom_models_file.exists():
            raise FileNotFoundError(
                f"Custom models file {custom_models_file} does not exist."
            )
    else:
        custom_models_file = ROOT / "custom_models.yaml"
    register_models(custom_models_file)

    caller = get_caller(model=model)
    message = LLMMessage(
        Role="user",
        Message=TEST_PROMPT,
    )
    response = caller.call([message], max_tokens=None)
    print(f"Test prompt: {TEST_PROMPT}")
    print(f"Response from {model}:\n{response.Message}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--custom-models-file", required=False, type=str, default=None)
    opt = parser.parse_args()
    main(opt)
