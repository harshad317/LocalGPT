"""
TriviaQA dataset (factoid QA; useful as a proxy for SimpleQA-style evaluations).
https://huggingface.co/datasets/trivia_qa
"""

import re

from datasets import load_dataset

from .common import Task


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^0-9a-z\s]", re.IGNORECASE)


def _normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.casefold()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TriviaQA(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["unfiltered"], "subset must be unfiltered"
        assert split in ["train", "validation"], "split must be train|validation"
        self.ds = load_dataset("trivia_qa", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"].rstrip() + "\n\nAnswer with a short phrase only. No explanation."
        answer = row["answer"]
        answer_value = answer.get("value", "")
        normalized_aliases = answer.get("normalized_aliases") or []
        if not normalized_aliases:
            normalized_value = answer.get("normalized_value", "") or answer_value
            normalized_aliases = [_normalize_answer(normalized_value)]
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_value},
            ],
            "normalized_aliases": normalized_aliases,
        }

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False
        pred = _normalize_answer(assistant_response)
        aliases = conversation.get("normalized_aliases") or []
        if aliases:
            return pred in {_normalize_answer(alias) for alias in aliases}
        ref = conversation["messages"][-1]["content"]
        return pred == _normalize_answer(ref)
