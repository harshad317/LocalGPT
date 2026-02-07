"""
XLAM function calling dataset (tool-use / function-call style).
https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k

Note: nanochat currently has special tokens only for the Python REPL tool. This task
represents function calls as plain JSON text in the assistant response (no tool parts).
This makes it usable for training and for text-parsing function calling evals (e.g. BFCL-style).
"""

import json
from datasets import load_dataset

from .common import Task

try:
    from .tool_calling_datasets import augment_tools, build_tool_prompt_messages, DEFAULT_TOOL_SCHEMA
    _TOOL_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    if exc.name not in ("tasks.tool_calling_datasets", "tool_calling_datasets"):
        raise
    augment_tools = None
    build_tool_prompt_messages = None
    DEFAULT_TOOL_SCHEMA = []
    _TOOL_IMPORT_ERROR = exc


class XLAMFunctionCalling(Task):
    def __init__(self, split, add_distractors=0, distractor_tools=None, use_system_prompt=True, **kwargs):
        super().__init__(**kwargs)
        if augment_tools is None or build_tool_prompt_messages is None:
            raise ModuleNotFoundError(
                "Missing tasks.tool_calling_datasets. Restore tasks/tool_calling_datasets.py or update your checkout."
            ) from _TOOL_IMPORT_ERROR
        assert split in ["train"], "XLAMFunctionCalling split must be train"
        self.ds = load_dataset("Salesforce/xlam-function-calling-60k", "dataset", split=split).shuffle(seed=42)
        self.add_distractors = max(0, int(add_distractors))
        self.distractor_tools = list(distractor_tools) if distractor_tools is not None else list(DEFAULT_TOOL_SCHEMA)
        self.use_system_prompt = use_system_prompt

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        query = row["query"]
        tools_json = row["tools"]
        answers_json = row["answers"]
        # Validate that these are JSON strings
        tools = json.loads(tools_json)
        answers = json.loads(answers_json)
        assert isinstance(tools, list), "Expected tools to be a JSON list"
        assert isinstance(answers, list), "Expected answers to be a JSON list"
        assert len(answers) >= 1, "Expected at least one answer"
        tools = augment_tools(tools, self.distractor_tools, self.add_distractors)
        assistant = json.dumps(answers, ensure_ascii=False)
        messages = build_tool_prompt_messages(tools, query, use_system_prompt=self.use_system_prompt)
        return {
            "messages": [
                *messages,
                {"role": "assistant", "content": assistant},
            ],
        }

    def evaluate(self, conversation, assistant_response):
        # Best-effort JSON validity check.
        if not isinstance(assistant_response, str):
            return False
        try:
            json.loads(assistant_response)
            return True
        except Exception:
            return False
