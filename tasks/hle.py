"""
Humanity's Last Exam (HLE) dataset.
https://huggingface.co/datasets/cais/hle
"""

import re
from datasets import load_dataset

from .common import Task


_CHOICE_RE = re.compile(r"\b([A-Z])\b")
_WS_RE = re.compile(r"\s+")


def _normalize_exact(text: str) -> str:
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    return text.casefold()


def _extract_choice(text: str):
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    match = _CHOICE_RE.search(stripped.upper())
    if match:
        return match.group(1)
    first = stripped[0].upper()
    if "A" <= first <= "Z":
        return first
    return None


class HLE(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["test"], "HLE split must be test"
        self.ds = load_dataset("cais/hle", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        answer_type = row["answer_type"]

        if answer_type == "multipleChoice":
            question = f"{question}\n\nRespond only with the letter of the correct answer."
        else:
            question = f"{question}\n\nRespond with the exact answer only."

        conversation = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "answer_type": answer_type,
        }
        if answer_type == "multipleChoice":
            conversation["correct_letter"] = answer.strip().upper()
        return conversation

    def evaluate(self, conversation, assistant_response):
        answer_type = conversation.get("answer_type", "exactMatch")
        if answer_type == "multipleChoice":
            expected = conversation.get("correct_letter")
            pred = _extract_choice(assistant_response)
            return pred is not None and pred == expected
        if not isinstance(assistant_response, str):
            return False
        expected = conversation["messages"][-1]["content"]
        return _normalize_exact(assistant_response) == _normalize_exact(expected)
