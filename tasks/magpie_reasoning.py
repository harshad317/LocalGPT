"""
Magpie Reasoning dataset.
https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V1-150K
"""

from datasets import load_dataset

from .common import Task
from .hf_utils import convert_conversations


class MagpieReasoning(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "MagpieReasoning split must be train"
        self.ds = load_dataset("Magpie-Align/Magpie-Reasoning-V1-150K", split=split).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        convs = row.get("conversations") or []
        if convs:
            messages = convert_conversations(convs, role_key="from", content_key="value")
        else:
            instruction = row.get("instruction") or ""
            response = row.get("response") or ""
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        return {"messages": messages}
