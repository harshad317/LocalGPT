"""
Natural Reasoning dataset.
https://huggingface.co/datasets/facebook/natural_reasoning
"""

from datasets import load_dataset

from .common import Task


class NaturalReasoning(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "NaturalReasoning split must be train"
        self.ds = load_dataset("facebook/natural_reasoning", split=split).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row.get("question") or ""
        reference = row.get("reference_answer") or ""
        if not reference:
            responses = row.get("responses") or []
            if responses:
                reference = responses[0].get("response") or ""
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reference},
            ],
        }
