import pytest

from tasks.common import Task, TaskMixture


class _Gen(Task):
    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return 1

    def get_example(self, index):
        return {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}


class _Cat(Task):
    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return 1

    def get_example(self, index):
        return {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "A"}], "letters": ("A", "B")}


def test_task_mixture_eval_type_uniform():
    m = TaskMixture([_Gen(), _Gen()])
    assert m.eval_type == "generative"


def test_task_mixture_eval_type_mixed_raises():
    m = TaskMixture([_Gen(), _Cat()])
    with pytest.raises(ValueError, match="mixed eval types"):
        _ = m.eval_type

