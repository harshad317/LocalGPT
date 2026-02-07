"""
Skunkworks reasoning dataset (small CoT sample).
https://huggingface.co/datasets/SkunkworksAI/reasoning-0.01
"""

from datasets import load_dataset

from .common import Task


class SkunkworksReasoning(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "SkunkworksReasoning split must be train"
        self.ds = load_dataset("SkunkworksAI/reasoning-0.01", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"]
        reasoning = (row.get("reasoning") or "").strip()
        output = (row.get("output") or "").strip()

        if reasoning and output:
            assistant = f"{reasoning}\n\n{output}"
        elif reasoning:
            assistant = reasoning
        else:
            assistant = output

        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": assistant},
            ],
        }
