"""
MBPP: Mostly Basic Programming Problems (coding SFT proxy for HumanEval).
https://huggingface.co/datasets/google-research-datasets/mbpp
"""

import re
from datasets import load_dataset

from nanochat.execution import execute_code
from .common import Task


class MBPP(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], "MBPP split must be train|validation|test"
        self.ds = load_dataset("google-research-datasets/mbpp", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["text"]
        code = row["code"]
        test_setup_code = row.get("test_setup_code") or ""
        test_list = row.get("test_list") or []
        challenge_test_list = row.get("challenge_test_list") or []
        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": code},
            ],
            "test_setup_code": test_setup_code,
            "test_list": test_list,
            "challenge_test_list": challenge_test_list,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False

        # Extract Python code from the completion (handles ```python blocks).
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, assistant_response, re.DOTALL)
        completion_code = matches[0].strip() if matches else assistant_response.strip()

        tests = []
        for t in conversation.get("test_list", []):
            if isinstance(t, str) and t.strip():
                tests.append(t.strip())
        for t in conversation.get("challenge_test_list", []):
            if isinstance(t, str) and t.strip():
                tests.append(t.strip())
        if not tests:
            return False

        setup = conversation.get("test_setup_code") or ""
        program = completion_code + "\n\n" + setup + "\n\n" + "\n".join(tests) + "\n"
        result = execute_code(program)
        return result.success
