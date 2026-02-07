"""
OpenMathInstruct-2 dataset (math reasoning).
https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
"""

from datasets import load_dataset

from .common import Task


class OpenMathInstruct2(Task):
    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "OpenMathInstruct2 split must be train"
        self.ds = load_dataset("nvidia/OpenMathInstruct-2", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["problem"]
        solution = (row.get("generated_solution") or "").strip()
        expected = (row.get("expected_answer") or "").strip()

        assistant = solution
        if expected and expected not in assistant:
            if assistant:
                assistant = f"{assistant}\n\nFinal answer: {expected}"
            else:
                assistant = expected

        return {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": assistant},
            ],
        }
