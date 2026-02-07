import json


class DummyDataset:
    def __init__(self, rows):
        self._rows = list(rows)

    def shuffle(self, seed=42):
        return self

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _assert_conversation(conversation):
    assert "messages" in conversation
    msgs = conversation["messages"]
    assert len(msgs) >= 2
    for i, m in enumerate(msgs):
        expected = "user" if i % 2 == 0 else "assistant"
        assert m["role"] == expected
        assert "content" in m


def test_hellaswag_formats_multiple_choice(monkeypatch):
    from tasks import hellaswag as mod

    rows = [
        {
            "ctx": "A person is cooking pasta.",
            "endings": ["They go for a walk.", "They add salt.", "They open a book.", "They wash a car."],
            "label": "1",
        }
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda *args, **kwargs: DummyDataset(rows))
    ds = mod.HellaSwag(split="train")
    conv = ds[0]
    _assert_conversation(conv)
    assert conv["messages"][-1]["content"] in ds.letters


def test_gpqa_formats_multiple_choice(monkeypatch):
    from tasks import gpqa as mod

    rows = [
        {
            "Question": "What is 2+2?",
            "Correct Answer": "4",
            "Incorrect Answer 1": "3",
            "Incorrect Answer 2": "5",
            "Incorrect Answer 3": "22",
        }
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda *args, **kwargs: DummyDataset(rows))
    ds = mod.GPQA(subset="gpqa_main", split="train")
    conv = ds[0]
    _assert_conversation(conv)
    assert conv["messages"][-1]["content"] in ds.letters


def test_hendrycks_math_formats_problem_solution(monkeypatch):
    from tasks import hendrycks_math as mod

    rows = [
        {"problem": "Compute 1+1.", "solution": "We get 2. \\\\boxed{2}", "type": "algebra", "level": "Level 1"}
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda *args, **kwargs: DummyDataset(rows))
    ds = mod.HendrycksMath(subject="algebra", split="train")
    conv = ds[0]
    _assert_conversation(conv)
    assert "\\\\boxed" in conv["messages"][-1]["content"]
    assert ds.evaluate(conv, "Answer: \\\\boxed{2}")


def test_mbpp_formats_code(monkeypatch):
    from tasks import mbpp as mod

    rows = [{"text": "Write a function add.", "code": "def add(a,b):\n    return a+b"}]
    monkeypatch.setattr(mod, "load_dataset", lambda *args, **kwargs: DummyDataset(rows))
    ds = mod.MBPP(split="train")
    conv = ds[0]
    _assert_conversation(conv)
    assert "def add" in conv["messages"][-1]["content"]


def test_alpaca_formats_instruction(monkeypatch):
    from tasks import alpaca as mod

    rows = [{"instruction": "Say hi", "input": "", "output": "Hi!"}]
    monkeypatch.setattr(mod, "load_dataset", lambda *args, **kwargs: DummyDataset(rows))
    ds = mod.Alpaca(split="train")
    conv = ds[0]
    _assert_conversation(conv)
    assert conv["messages"][0]["content"] == "Say hi"
    assert conv["messages"][-1]["content"] == "Hi!"


def test_xlam_function_calling_emits_json(monkeypatch):
    from tasks import xlam_function_calling as mod

    rows = [
        {
            "query": "What time is it in NYC?",
            "tools": json.dumps([{"name": "get_time", "parameters": {"type": "object", "properties": {}}}]),
            "answers": json.dumps([{"name": "get_time", "arguments": {"city": "NYC"}}]),
        }
    ]
    monkeypatch.setattr(mod, "load_dataset", lambda *args, **kwargs: DummyDataset(rows))
    ds = mod.XLAMFunctionCalling(split="train")
    conv = ds[0]
    _assert_conversation(conv)
    parsed = json.loads(conv["messages"][-1]["content"])
    assert parsed[0]["name"] == "get_time"


def test_triviaqa_formats_factoid(monkeypatch):
    from tasks import triviaqa as mod

    rows = [{"question": "Capital of France?", "answer": {"value": "Paris"}}]
    monkeypatch.setattr(mod, "load_dataset", lambda *args, **kwargs: DummyDataset(rows))
    ds = mod.TriviaQA(subset="unfiltered", split="train")
    conv = ds[0]
    _assert_conversation(conv)
    assert conv["messages"][-1]["content"] == "Paris"

