"""
HellaSwag dataset (commonsense multiple choice).
https://huggingface.co/datasets/hellaswag
"""

import random
from datasets import load_dataset

from .common import Task, render_mc


class HellaSwag(Task):
    letters = ("A", "B", "C", "D")

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation"], "HellaSwag split must be train|validation"
        self.ds = load_dataset("hellaswag", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        context = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        assert len(endings) == 4, "HellaSwag should have exactly 4 endings"
        assert 0 <= label < 4, f"Invalid HellaSwag label {label}"

        # Deterministically shuffle choices so the correct option isn't always in the same position.
        rng = random.Random(index)
        order = list(range(4))
        rng.shuffle(order)
        shuffled_endings = [endings[i] for i in order]
        new_label = order.index(label)

        question = (
            "Choose the best continuation for the following context.\n\n"
            f"Context: {context}\n\n"
            "Which option is the most likely next text?"
        )
        user_message = render_mc(question, self.letters, shuffled_endings)
        assistant_message = self.letters[new_label]
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ],
            "letters": self.letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"HellaSwag answer {assistant_response} is expected to be one of {conversation['letters']}"
        )
        ref = conversation["messages"][-1]["content"]
        return assistant_response == ref
