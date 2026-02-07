"""
Code Bagel dataset (instruction-following code tasks).
https://huggingface.co/datasets/Replete-AI/code_bagel
"""

from datasets import load_dataset

from .common import Task


class CodeBagel(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "CodeBagel split must be train"
        self.ds = load_dataset("Replete-AI/code_bagel", split=split).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        instruction = row.get("instruction") or ""
        input_text = row.get("input") or ""
        output_text = row.get("output") or ""

        if input_text.strip():
            user_message = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_message = instruction

        return {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": output_text},
            ],
        }
