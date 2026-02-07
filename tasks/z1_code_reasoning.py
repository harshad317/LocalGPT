"""
Z1 Code Reasoning dataset.
https://huggingface.co/datasets/efficientscaling/Z1-Code-Reasoning-107K
"""

from datasets import load_dataset

from .common import Task


class Z1CodeReasoning(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "Z1CodeReasoning split must be train"
        self.ds = load_dataset("efficientscaling/Z1-Code-Reasoning-107K", split=split).shuffle(seed=42)
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
        response = row.get("response") or ""
        return {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ],
        }
