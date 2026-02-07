"""
BFCL v3 (Berkeley Function-Calling Leaderboard) benchmark dataset.
Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
"""

import ast
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from datasets import Dataset

from .common import Task, TaskMixture

__all__ = ["BFCLV3", "build_bfcl_v3_benchmark"]


BFCL_V3_COMMIT = os.environ.get(
    "BFCL_V3_COMMIT",
    "70b6a4a2144597b1f99d1f4d3185d35d7ee532a4",
)
BFCL_V3_DATA_BASE_URL = os.environ.get(
    "BFCL_V3_DATA_BASE_URL",
    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_V3_COMMIT}/berkeley-function-call-leaderboard/data",
)

BFCL_V3_TEST_FILE_MAPPING = {
    "exec_simple": "BFCL_v3_exec_simple.json",
    "exec_parallel": "BFCL_v3_exec_parallel.json",
    "exec_multiple": "BFCL_v3_exec_multiple.json",
    "exec_parallel_multiple": "BFCL_v3_exec_parallel_multiple.json",
    "simple": "BFCL_v3_simple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
    "parallel": "BFCL_v3_parallel.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "java": "BFCL_v3_java.json",
    "javascript": "BFCL_v3_javascript.json",
    "rest": "BFCL_v3_rest.json",
    "sql": "BFCL_v3_sql.json",
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
    "live_parallel": "BFCL_v3_live_parallel.json",
    "live_parallel_multiple": "BFCL_v3_live_parallel_multiple.json",
    "live_irrelevance": "BFCL_v3_live_irrelevance.json",
    "live_relevance": "BFCL_v3_live_relevance.json",
    "multi_turn_base": "BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context.json",
    "multi_turn_composite": "BFCL_v3_multi_turn_composite.json",
}

BFCL_V3_COLLECTIONS = {
    "all": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "rest",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
        "multi_turn_composite",
    ],
    "multi_turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
        "multi_turn_composite",
    ],
    "single_turn": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "rest",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "live": [
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "non_live": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "java",
        "javascript",
        "rest",
    ],
    "executable": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "rest",
    ],
    "non_python": [
        "java",
        "javascript",
    ],
    "python": [
        "exec_simple",
        "exec_parallel",
        "exec_multiple",
        "exec_parallel_multiple",
        "simple",
        "irrelevance",
        "parallel",
        "multiple",
        "parallel_multiple",
        "rest",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
}

BFCL_V3_ANSWER_SUBSETS = {
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "java",
    "javascript",
    "sql",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "multi_turn_base",
    "multi_turn_composite",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
}


def _dataset_url(filename: str) -> str:
    return f"{BFCL_V3_DATA_BASE_URL}/{filename}"


def _cached_path(url: str) -> Optional[str]:
    try:
        from datasets.utils.file_utils import get_from_cache
    except Exception:
        return None
    try:
        return get_from_cache(url)
    except Exception:
        return None


def _parse_json_payload(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        rows = []
        for line in payload.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def _read_json_url(url: str) -> Any:
    cached = _cached_path(url)
    if cached and os.path.exists(cached):
        with open(cached, "r", encoding="utf-8") as handle:
            payload = handle.read()
        return _parse_json_payload(payload)
    with urllib.request.urlopen(url) as handle:
        payload = handle.read().decode("utf-8")
    return _parse_json_payload(payload)


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


def _decode_raw_row(row: Any) -> Any:
    if isinstance(row, dict) and isinstance(row.get("raw"), str):
        try:
            return json.loads(row["raw"])
        except Exception:
            return row
    return row


def _load_json_dataset(url: str):
    payload = _read_json_url(url)
    rows = _as_list(payload)
    try:
        return Dataset.from_list(rows)
    except Exception:
        raw_rows = [{"raw": json.dumps(row, ensure_ascii=False)} for row in rows]
        return Dataset.from_list(raw_rows)


def _ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _ast_name(node.value) + "." + node.attr
    return ""


def _parse_call_string(call_str: str) -> Optional[Dict[str, Any]]:
    try:
        node = ast.parse(call_str, mode="eval").body
    except Exception:
        return None
    if not isinstance(node, ast.Call):
        return None
    name = _ast_name(node.func)
    args: Dict[str, Any] = {}
    if node.args:
        try:
            args["_args"] = [ast.literal_eval(arg) for arg in node.args]
        except Exception:
            return None
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return None
    return {"name": name, "arguments": args}


def _expected_from_dict(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not entry:
        return None
    if len(entry) != 1:
        # Unexpected structure; fall back to the first key.
        name = next(iter(entry))
        params = entry[name]
    else:
        name, params = next(iter(entry.items()))
    if not isinstance(params, dict):
        return None
    normalized = {}
    for key, values in params.items():
        if isinstance(values, list):
            normalized[key] = values
        else:
            normalized[key] = [values]
    return {"name": name, "arguments": normalized}


def _expected_from_call(call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not call or "name" not in call:
        return None
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return None
    normalized = {}
    for key, value in args.items():
        normalized[key] = [value]
    return {"name": call["name"], "arguments": normalized}


def _expected_from_raw(raw_gt: Any) -> Optional[Any]:
    if raw_gt is None:
        return None
    if isinstance(raw_gt, list):
        if raw_gt and isinstance(raw_gt[0], list):
            return [_expected_from_raw(turn_gt) for turn_gt in raw_gt]
        if not raw_gt:
            return []
        if all(isinstance(item, dict) for item in raw_gt):
            expected = []
            for item in raw_gt:
                parsed = _expected_from_dict(item)
                if parsed is None:
                    return None
                expected.append(parsed)
            return expected
        if all(isinstance(item, str) for item in raw_gt):
            expected = []
            for item in raw_gt:
                parsed = _parse_call_string(item)
                if parsed is None:
                    return None
                parsed = _expected_from_call(parsed)
                if parsed is None:
                    return None
                expected.append(parsed)
            return expected
    if isinstance(raw_gt, dict):
        parsed = _expected_from_dict(raw_gt)
        return [parsed] if parsed is not None else None
    if isinstance(raw_gt, str):
        parsed = _parse_call_string(raw_gt)
        parsed = _expected_from_call(parsed) if parsed else None
        return [parsed] if parsed is not None else None
    return None


def _pick_canonical_value(values: List[Any]) -> Any:
    for value in values:
        if value not in ("", None):
            return value
    return values[0] if values else None


def _expected_to_calls(expected: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not expected:
        return []
    calls = []
    for exp in expected:
        if not exp or "name" not in exp:
            continue
        args = {}
        for key, values in exp.get("arguments", {}).items():
            if not isinstance(values, list):
                values = [values]
            value = _pick_canonical_value(values)
            if value in ("", None):
                continue
            args[key] = value
        calls.append({"name": exp["name"], "arguments": args})
    return calls


def _json_dumps_calls(calls: List[Dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False)


def _normalize_question(raw_question: Any) -> List[str]:
    if isinstance(raw_question, str):
        return [raw_question]
    if isinstance(raw_question, list):
        if raw_question and isinstance(raw_question[0], dict):
            raw_question = [raw_question]
        turns = []
        for turn in raw_question:
            if isinstance(turn, dict):
                turns.append(str(turn.get("content", "")))
            elif isinstance(turn, list):
                parts = [str(msg.get("content", "")) for msg in turn if isinstance(msg, dict)]
                turns.append("\n\n".join(part for part in parts if part).strip())
            else:
                turns.append(str(turn))
        return turns
    return [str(raw_question)]


def _normalize_predicted_calls(payload: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return None
    calls = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        name = item.get("name") or item.get("function")
        if not name:
            return None
        args = item.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return None
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None
        calls.append({"name": name, "arguments": args})
    return calls


def _coerce_number(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        try:
            if "." in text:
                return float(text)
            return int(text)
        except Exception:
            return value
    return value


def _value_matches(pred: Any, allowed_values: List[Any]) -> bool:
    pred_norm = _coerce_number(pred)
    for allowed in allowed_values:
        allowed_norm = _coerce_number(allowed)
        if pred_norm == allowed_norm:
            return True
    return False


def _call_matches(pred_call: Dict[str, Any], exp_call: Dict[str, Any]) -> bool:
    if pred_call.get("name") != exp_call.get("name"):
        return False
    pred_args = pred_call.get("arguments", {}) or {}
    exp_args = exp_call.get("arguments", {}) or {}
    for key, allowed in exp_args.items():
        if not isinstance(allowed, list):
            allowed = [allowed]
        if key not in pred_args:
            if "" in allowed or None in allowed:
                continue
            return False
        if not _value_matches(pred_args.get(key), allowed):
            return False
    return True


def _match_expected(pred_calls: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> bool:
    if len(pred_calls) != len(expected):
        return False
    remaining = list(pred_calls)
    for exp_call in expected:
        match_idx = None
        for idx, pred_call in enumerate(remaining):
            if _call_matches(pred_call, exp_call):
                match_idx = idx
                break
        if match_idx is None:
            return False
        remaining.pop(match_idx)
    return True


class BFCLV3(Task):
    def __init__(self, subset: str, split: str = "test", **kwargs):
        super().__init__(**kwargs)
        if subset not in BFCL_V3_TEST_FILE_MAPPING:
            known = ", ".join(sorted(BFCL_V3_TEST_FILE_MAPPING))
            raise ValueError(f"Unknown BFCL v3 subset '{subset}'. Available: {known}")
        if split not in ["test", "train"]:
            raise ValueError("BFCLV3 split must be test|train (dataset is test-only)")
        self.subset = subset
        self.split = split
        filename = BFCL_V3_TEST_FILE_MAPPING[subset]
        self.ds = _load_json_dataset(_dataset_url(filename)).shuffle(seed=42)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)

        self.answer_map: Dict[str, Any] = {}
        if subset in BFCL_V3_ANSWER_SUBSETS:
            answer_url = _dataset_url(f"possible_answer/{filename}")
            answers = _load_json_dataset(answer_url)
            for row in answers:
                row = _decode_raw_row(row)
                if not isinstance(row, dict):
                    continue
                row_id = row.get("id")
                if row_id is None:
                    continue
                self.answer_map[row_id] = row.get("ground_truth")

        self.force_empty_if_missing = "irrelevance" in subset

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def _build_expected_per_turn(self, raw_gt: Any, num_turns: int) -> List[Optional[List[Dict[str, Any]]]]:
        expected = _expected_from_raw(raw_gt)
        if expected is None:
            return [None] * num_turns
        if isinstance(expected, list) and expected and isinstance(expected[0], list):
            expected_turns = expected
        else:
            expected_turns = [None] * max(0, num_turns - 1) + [expected]
        if len(expected_turns) < num_turns:
            expected_turns = expected_turns + [None] * (num_turns - len(expected_turns))
        if len(expected_turns) > num_turns:
            expected_turns = expected_turns[:num_turns]
        return expected_turns

    def get_example(self, index):
        row = _decode_raw_row(self.ds[index])
        if not isinstance(row, dict):
            raise ValueError("BFCL v3 row has unsupported schema.")
        row_id = row.get("id") or f"{self.subset}_{index}"
        functions = row.get("function")
        if not isinstance(functions, list):
            raise ValueError(f"BFCL v3 subset '{self.subset}' has unsupported function schema.")

        raw_question = row.get("question")
        turns = _normalize_question(raw_question)
        if not turns:
            raise ValueError("BFCL v3 question is empty")

        raw_gt = row.get("ground_truth")
        if raw_gt is None and row_id in self.answer_map:
            raw_gt = self.answer_map[row_id]
        if raw_gt is None and self.force_empty_if_missing:
            raw_gt = []

        expected_turns = self._build_expected_per_turn(raw_gt, len(turns))

        tools_json = json.dumps(functions, ensure_ascii=False)
        system_prompt = (
            "You have access to the following tools (JSON schema):\n"
            f"{tools_json}\n\n"
            "Respond with a JSON array of tool calls only, e.g. "
            '[{"name": "...", "arguments": {...}}]. If no tool is needed, respond with [].'
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn, expected in zip(turns, expected_turns):
            messages.append({"role": "user", "content": turn})
            calls = _expected_to_calls(expected)
            messages.append({"role": "assistant", "content": _json_dumps_calls(calls)})

        conversation = {
            "messages": messages,
            "bfcl_expected_calls": expected_turns[-1],
            "bfcl_expect_empty": expected_turns[-1] == [] if expected_turns else False,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False
        try:
            payload = json.loads(assistant_response)
        except Exception:
            return False
        predicted_calls = _normalize_predicted_calls(payload)
        if predicted_calls is None:
            return False

        expected = conversation.get("bfcl_expected_calls")
        if expected is None:
            if conversation.get("bfcl_expect_empty"):
                return len(predicted_calls) == 0
            return True
        return _match_expected(predicted_calls, expected)


def build_bfcl_v3_benchmark(collection: str = "all", **kwargs) -> TaskMixture:
    if collection not in BFCL_V3_COLLECTIONS:
        known = ", ".join(sorted(BFCL_V3_COLLECTIONS))
        raise ValueError(f"Unknown BFCL v3 collection '{collection}'. Available: {known}")
    subsets = BFCL_V3_COLLECTIONS[collection]
    return TaskMixture([BFCLV3(subset=subset, **kwargs) for subset in subsets])
