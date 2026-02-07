"""
American Invitational Mathematics Examination (AIME) datasets.
"""

import re
from datasets import load_dataset

from .common import Task


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_FRAMEBOX_RE = re.compile(r"\\framebox\{([^}]*)\}")
_INT_RE = re.compile(r"\d+")


def _extract_int(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    for pattern in (_BOXED_RE, _FRAMEBOX_RE):
        match = pattern.search(text)
        if match:
            digits = _INT_RE.findall(match.group(1))
            if digits:
                return int(digits[-1])
    digits = _INT_RE.findall(text)
    if digits:
        return int(digits[-1])
    return None


class _AIMEBase(Task):
    dataset_name = ""
    year = None

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "AIME dataset supports only train split"
        self.split = split
        self.ds = load_dataset(self.dataset_name, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["problem"].rstrip()
        prompt = f"{problem}\n\nProvide the final answer as an integer between 0 and 999."
        solution = row.get("solution") or row.get("answer") or ""
        conversation = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": solution},
            ],
            "answer": row.get("answer"),
            "id": row.get("id"),
            "url": row.get("url"),
            "year": row.get("year", self.year),
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False
        ref_answer = conversation.get("answer")
        if ref_answer is None:
            ref_answer = conversation["messages"][-1]["content"]
        ref_int = _extract_int(str(ref_answer))
        pred_int = _extract_int(assistant_response)
        if ref_int is None or pred_int is None:
            return False
        return ref_int == pred_int


class AIME2024(_AIMEBase):
    dataset_name = "OpenEvals/aime_24"
    year = "2024"


class AIME2025(_AIMEBase):
    dataset_name = "yentinglin/aime_2025"
    year = 2025
