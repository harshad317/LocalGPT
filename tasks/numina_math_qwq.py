"""
NuminaMath-QwQ-CoT dataset.
https://huggingface.co/datasets/PrimeIntellect/NuminaMath-QwQ-CoT-5M
"""

from datasets import load_dataset

from .common import Task


class NuminaMathQwQ(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "NuminaMathQwQ split must be train"
        self.ds = load_dataset("PrimeIntellect/NuminaMath-QwQ-CoT-5M", split=split).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row.get("prompt") or ""
        response = row.get("response") or ""
        ground_truth = row.get("ground_truth") or ""
        correct = bool(row.get("correct"))
        assistant = response if (correct and response) else ground_truth
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant},
            ],
            "ground_truth": ground_truth,
            "correct": correct,
        }
