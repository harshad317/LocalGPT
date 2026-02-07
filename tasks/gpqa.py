"""
GPQA dataset (graduate-level multiple choice questions).
https://huggingface.co/datasets/Idavidrein/gpqa
"""

import random
from datasets import load_dataset

from .common import Task, render_mc


class GPQA(Task):
    letters = ("A", "B", "C", "D")

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["gpqa_main", "gpqa_extended", "gpqa_diamond", "gpqa_experts"], (
            "GPQA subset must be gpqa_main|gpqa_extended|gpqa_diamond|gpqa_experts"
        )
        assert split in ["train"], "GPQA split must be train"
        self.ds = load_dataset("Idavidrein/gpqa", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["Question"]
        correct = row["Correct Answer"]
        incorrect = [row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        choices = [correct] + incorrect

        rng = random.Random(index)
        order = list(range(4))
        rng.shuffle(order)
        shuffled_choices = [choices[i] for i in order]
        correct_letter = self.letters[order.index(0)]

        user_message = render_mc(question, self.letters, shuffled_choices)
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": correct_letter},
            ],
            "letters": self.letters,
            "correct_letter": correct_letter,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"GPQA answer {assistant_response} is expected to be one of {conversation['letters']}"
        )
        return assistant_response == conversation["correct_letter"]
