"""
MMLU-Pro dataset (harder MMLU variant with more options).
https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
"""

from datasets import load_dataset

from .common import Task, render_mc


class MMLUPro(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["validation", "test"], "MMLUPro split must be validation|test"
        self.ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        options = row["options"]
        answer_index = int(row["answer_index"])

        assert isinstance(options, list) and len(options) >= 2, "MMLUPro options must be a non-empty list"
        assert 0 <= answer_index < len(options), f"Invalid answer_index {answer_index} for {len(options)} options"

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert len(options) <= len(alphabet), f"Too many options ({len(options)}), max is {len(alphabet)}"
        letters = tuple(alphabet[: len(options)])

        user_message = render_mc(question, letters, options)
        assistant_message = letters[answer_index]
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            "letters": letters,
            "category": row.get("category", None),
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"MMLUPro answer {assistant_response} is expected to be one of {conversation['letters']}"
        )
        return assistant_response == conversation["messages"][-1]["content"]
