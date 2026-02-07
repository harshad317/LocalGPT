"""
Lightweight STEM QA datasets (non-coding).
"""

import hashlib
import random

from datasets import load_dataset

try:
    from .common import Task, render_mc
except ImportError:
    import tasks.common as _tc

    Task = getattr(_tc, "Task", None) or getattr(_tc, "BaseTask", None) or getattr(_tc, "TaskBase", None)
    if Task is None:
        class Task:  # fallback minimal Task
            def __init__(self, start=0, stop=None, step=1):
                self.start = start
                self.stop = stop
                self.step = step

            @property
            def eval_type(self):
                raise NotImplementedError

            def num_examples(self):
                raise NotImplementedError

            def get_example(self, index):
                raise NotImplementedError

            def __len__(self):
                start = self.start
                stop = self.num_examples() if self.stop is None else self.stop
                step = self.step
                span = stop - start
                return (span + step - 1) // step

            def __getitem__(self, index: int):
                physical_index = self.start + index * self.step
                return self.get_example(physical_index)

            def evaluate(self, problem, completion):
                raise NotImplementedError

    render_mc = getattr(_tc, "render_mc", None)
    if render_mc is None:
        def render_mc(question, letters, choices):
            query = f"Multiple Choice question: {question}\n"
            query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
            query += "\nRespond only with the letter of the correct answer."
            return query


def _stable_seed(text):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


def _shuffle_with_key(choices, key_text):
    rng = random.Random(_stable_seed(key_text))
    shuffled = list(choices)
    rng.shuffle(shuffled)
    return shuffled


def _mc_conversation(question, choices, correct_index):
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    user_message = render_mc(question, letters, choices)
    assistant_message = letters[correct_index]
    conversation = {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "letters": letters,
    }
    return conversation


class SciQ(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], "SciQ split must be train|validation|test"
        self.ds = load_dataset("allenai/sciq", split=split).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row.get("question") or ""
        correct = row.get("correct_answer")
        distractors = [
            row.get("distractor1"),
            row.get("distractor2"),
            row.get("distractor3"),
        ]
        if not question or correct is None:
            raise ValueError("SciQ row missing question/correct_answer")
        choices = [(correct, True)] + [(d, False) for d in distractors if d]
        choices = _shuffle_with_key(choices, f"{question}||{correct}")
        choice_texts = [text for text, _ in choices]
        correct_index = next(i for i, (_, is_correct) in enumerate(choices) if is_correct)
        return _mc_conversation(question, choice_texts, correct_index)

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"SciQ answer {assistant_response} is expected to be one of {conversation['letters']}"
        )
        return assistant_response == conversation["messages"][-1]["content"]


class OpenBookQA(Task):
    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], "OpenBookQA split must be train|validation|test"
        self.ds = load_dataset("allenai/openbookqa", split=split).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def _parse_choices(self, choices_raw):
        if isinstance(choices_raw, dict) and "text" in choices_raw and "label" in choices_raw:
            texts = list(choices_raw["text"])
            labels = list(choices_raw["label"])
            return labels, texts
        if isinstance(choices_raw, list):
            labels = []
            texts = []
            for item in choices_raw:
                if not isinstance(item, dict):
                    continue
                labels.append(item.get("label"))
                texts.append(item.get("text"))
            if labels and texts:
                pairs = sorted(zip(labels, texts), key=lambda x: str(x[0]))
                labels = [p[0] for p in pairs]
                texts = [p[1] for p in pairs]
                return labels, texts
        raise ValueError("OpenBookQA row missing choices")

    def get_example(self, index):
        row = self.ds[index]
        question = row.get("question_stem") or row.get("question") or ""
        answer_key = row.get("answerKey") or row.get("answer") or row.get("label")
        if not question or answer_key is None:
            raise ValueError("OpenBookQA row missing question/answer")
        labels, choices = self._parse_choices(row.get("choices"))
        if answer_key not in labels:
            if str(answer_key).isdigit():
                idx = int(answer_key) - 1
                if 0 <= idx < len(labels):
                    answer_key = labels[idx]
        if answer_key not in labels:
            raise ValueError("OpenBookQA answerKey not found in labels")
        user_message = render_mc(question, labels, choices)
        conversation = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer_key},
            ],
            "letters": labels,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"OpenBookQA answer {assistant_response} is expected to be one of {conversation['letters']}"
        )
        return assistant_response == conversation["messages"][-1]["content"]
