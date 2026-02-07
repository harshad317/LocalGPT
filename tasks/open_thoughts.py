"""
OpenThoughts2 reasoning dataset.
https://huggingface.co/datasets/open-thoughts/OpenThoughts2-1M
"""

from datasets import load_dataset

from .common import Task
from .hf_utils import convert_conversations


class OpenThoughts2(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "OpenThoughts2 split must be train"
        self.ds = load_dataset("open-thoughts/OpenThoughts2-1M", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = convert_conversations(row["conversations"], role_key="from", content_key="value")
        return {"messages": messages}
