"""
Alpaca instruction-following dataset (general instruction tuning).
https://huggingface.co/datasets/tatsu-lab/alpaca
"""

from datasets import load_dataset

from .common import Task


class Alpaca(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "Alpaca split must be train"
        self.ds = load_dataset("tatsu-lab/alpaca", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"]
        input_text = row.get("input") or ""
        output_text = row["output"]

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

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False
        return assistant_response.strip() == conversation["messages"][-1]["content"].strip()
