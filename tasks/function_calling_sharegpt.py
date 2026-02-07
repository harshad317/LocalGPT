"""
Function-calling ShareGPT dataset.
https://huggingface.co/datasets/hypervariance/function-calling-sharegpt
"""

from datasets import load_dataset

from .common import Task
from .hf_utils import convert_conversations


class FunctionCallingShareGPT(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "FunctionCallingShareGPT split must be train"
        self.ds = load_dataset("hypervariance/function-calling-sharegpt", split=split).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = convert_conversations(row["conversations"], role_key="from", content_key="value")
        return {"messages": messages}
