"""
Hermes function calling dataset.
https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
"""

import json
import os
import urllib.request
from typing import Any, List

from datasets import Dataset, load_dataset

from .common import Task
from .hf_utils import convert_conversations

_HERMES_FILES = {
    "func-calling-singleturn": "func-calling-singleturn.json",
    "glaive-function-calling-5k": "glaive-function-calling-5k.json",
}


def _cached_path(url: str) -> str:
    try:
        from datasets.utils.file_utils import get_from_cache
    except Exception:
        return ""
    try:
        return get_from_cache(url)
    except Exception:
        return ""


def _read_json_url(url: str) -> Any:
    cached = _cached_path(url)
    if cached and os.path.exists(cached):
        with open(cached, "r", encoding="utf-8") as handle:
            return json.load(handle)
    with urllib.request.urlopen(url) as handle:
        payload = handle.read().decode("utf-8")
    return json.loads(payload)


def _as_list(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def _load_hermes_dataset(url: str) -> Dataset:
    payload = _read_json_url(url)
    rows = _as_list(payload)
    filtered = []
    for row in rows:
        if isinstance(row, dict) and "conversations" in row:
            filtered.append({"conversations": row["conversations"]})
    if filtered:
        return Dataset.from_list(filtered)
    # Fallback: let datasets parse if structure is unexpected.
    return load_dataset("json", data_files={"train": url}, split="train")


class HermesFunctionCalling(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "HermesFunctionCalling split must be train"
        assert subset in _HERMES_FILES, f"Unknown Hermes subset {subset}"
        filename = _HERMES_FILES[subset]
        data_file = (
            "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1"
            f"/resolve/main/{filename}"
        )
        self.ds = _load_hermes_dataset(data_file).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = convert_conversations(row["conversations"], role_key="from", content_key="value")
        return {"messages": messages}
