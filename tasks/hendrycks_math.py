"""
Hendrycks Math dataset (competition-style math problems).
https://huggingface.co/datasets/EleutherAI/hendrycks_math
"""

import re
from datasets import load_dataset

from .common import Task


BOXED_RE = re.compile(r"\\+boxed\{([^}]*)\}")


def _extract_boxed(text: str):
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


class HendrycksMath(Task):
    def __init__(self, subject, split, **kwargs):
        super().__init__(**kwargs)
        subjects = [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
        assert subject in subjects, f"subject must be one of {subjects}"
        assert split in ["train", "test"], "split must be train|test"
        self.subject = subject
        self.ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["problem"].rstrip() + "\n\nProvide the final answer in the form \\boxed{...}."
        solution = row["solution"]
        conversation = {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
            "subject": self.subject,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False
        ref = conversation["messages"][-1]["content"]
        ref_boxed = _extract_boxed(ref)
        pred_boxed = _extract_boxed(assistant_response)
        if ref_boxed is not None and pred_boxed is not None:
            return ref_boxed == pred_boxed
        return assistant_response.strip() == ref.strip()
