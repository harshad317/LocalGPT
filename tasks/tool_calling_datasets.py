"""
Tool-calling dataset adapters for BFCL-style training (JSON-only tool calls).
"""

import json
import re
from typing import Any, Iterable, List, Optional, Tuple

from datasets import load_dataset

from .common import Task

DEFAULT_TOOL_SCHEMA = [
    {
        "name": "search_web",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current time for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "lookup_stock",
        "description": "Look up a stock price for a ticker.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
]

_TOOL_PROMPT_INSTRUCTION = (
    "Respond with a JSON array of tool calls only, e.g. "
    "[{\"name\": \"...\", \"arguments\": {...}}]. If no tool is needed, respond with []."
)

_TOOL_KEYS = (
    "tools",
    "tool",
    "functions",
    "function",
    "apis",
    "api",
    "tool_schema",
    "tool_schemas",
    "tool_spec",
    "tool_specs",
    "tool_specification",
    "tool_definitions",
    "tool_definition",
    "tool_list",
    "api_list",
    "api_specs",
    "api_spec",
    "schema",
    "schemas",
)

_QUERY_KEYS = (
    "query",
    "question",
    "instruction",
    "input",
    "prompt",
    "user_query",
    "text",
    "task",
    "goal",
    "utterance",
)

_CALL_KEYS = (
    "answers",
    "answer",
    "tool_calls",
    "tool_call",
    "function_calls",
    "function_call",
    "api_calls",
    "api_call",
    "calls",
    "call",
    "response",
    "output",
    "target",
    "completion",
)

_NO_TOOL_KEYS = (
    "needs_tool",
    "tool_required",
    "tool_needed",
    "requires_tool",
    "use_tool",
)

_ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "human": "user",
    "instruction": "user",
    "prompt": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "bot": "assistant",
    "response": "assistant",
    "tool": "user",
    "function": "user",
    "function_response": "user",
    "tool_response": "user",
}


def _parse_json_maybe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text[0] in "[{":
            try:
                return json.loads(text)
            except Exception:
                return None
    return None


def _extract_first(row: dict, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _is_english_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return True
    for key in ("lang", "language", "languages", "locale", "language_code"):
        val = row.get(key)
        if val is None:
            continue
        tokens = []
        if isinstance(val, (list, tuple)):
            tokens = [str(v).lower() for v in val]
        else:
            tokens = [str(val).lower()]
        for token in tokens:
            parts = re.split(r"[\s,;/|]+", token)
            for part in parts:
                if part.startswith("en") or part == "eng" or "english" in part:
                    return True
    return True


def _normalize_role(role_raw: Any, is_first: bool) -> str:
    role = str(role_raw).strip().lower()
    mapped = _ROLE_ALIASES.get(role)
    if mapped == "system":
        return "system" if is_first else "user"
    if mapped in ("user", "assistant"):
        return mapped
    return "user"


def _normalize_messages(messages: List[dict]) -> List[dict]:
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if "role" in msg and "content" in msg:
            role = msg.get("role")
            content = msg.get("content")
        elif "from" in msg and "value" in msg:
            role = msg.get("from")
            content = msg.get("value")
        else:
            continue
        normalized.append({
            "role": _normalize_role(role, is_first=len(normalized) == 0),
            "content": "" if content is None else str(content),
        })
    return normalized


def _extract_query_from_messages(raw: Any) -> Optional[str]:
    if not isinstance(raw, list):
        return None
    messages = _normalize_messages(raw)
    for msg in messages:
        if msg["role"] == "user" and msg["content"].strip():
            return msg["content"].strip()
    return None


def _extract_query(row: dict) -> Optional[str]:
    query = _extract_first(row, _QUERY_KEYS)
    if query is not None:
        return str(query)
    raw_messages = _extract_first(row, ("messages", "conversation", "conversations", "dialogue"))
    query = _extract_query_from_messages(raw_messages)
    if query:
        return query
    return None


def _normalize_params(params: Any) -> dict:
    if params is None:
        return {"type": "object", "properties": {}}
    if isinstance(params, str):
        parsed = _parse_json_maybe(params)
        if isinstance(parsed, dict):
            return parsed
        return {"type": "object", "properties": {"value": {"type": "string"}}}
    if isinstance(params, dict):
        return params
    return {"type": "object", "properties": {"value": {"type": "string"}}}


def _normalize_tool(tool: Any) -> Optional[dict]:
    if isinstance(tool, str):
        name = tool.strip()
        if not name:
            return None
        return {"name": name, "description": "", "parameters": {"type": "object", "properties": {}}}
    if isinstance(tool, dict):
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tool = tool["function"]
        name = tool.get("name") or tool.get("tool_name") or tool.get("function_name") or tool.get("api_name")
        if not name:
            return None
        desc = tool.get("description") or tool.get("desc") or tool.get("api_description") or ""
        params = tool.get("parameters") or tool.get("args_schema") or tool.get("input_schema") or tool.get("schema")
        params = _normalize_params(params)
        return {"name": str(name), "description": str(desc), "parameters": params}
    return None


def _coerce_tool_list(raw: Any) -> Optional[List[dict]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        parsed = _parse_json_maybe(raw)
        if parsed is None:
            return None
        raw = parsed
    if isinstance(raw, dict):
        for key in ("tools", "functions", "function", "apis", "api", "tool_list"):
            if key in raw:
                raw = raw[key]
                break
    if isinstance(raw, dict):
        tool = _normalize_tool(raw)
        return [tool] if tool else None
    if isinstance(raw, list):
        tools = []
        for item in raw:
            tool = _normalize_tool(item)
            if tool:
                tools.append(tool)
        return tools if tools else None
    return None


def _extract_tools(row: dict) -> Optional[List[dict]]:
    for key in _TOOL_KEYS:
        if key in row:
            tools = _coerce_tool_list(row[key])
            if tools:
                return tools
    name = _extract_first(row, ("tool_name", "api_name", "function_name", "name"))
    if name is not None:
        desc = _extract_first(row, ("tool_description", "api_description", "description", "desc"))
        params = _extract_first(row, ("parameters", "params", "args", "arguments", "input_schema", "schema"))
        tool = {
            "name": str(name),
            "description": "" if desc is None else str(desc),
            "parameters": _normalize_params(params),
        }
        return [tool]
    return None


def _normalize_arguments(args: Any) -> dict:
    if args is None:
        return {}
    if isinstance(args, str):
        parsed = _parse_json_maybe(args)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"args": parsed}
        return {"value": args}
    if isinstance(args, dict):
        return args
    if isinstance(args, list):
        return {"args": args}
    return {"value": args}


def _normalize_call(call: Any) -> Optional[dict]:
    if isinstance(call, str):
        parsed = _parse_json_maybe(call)
        if parsed is None:
            return None
        call = parsed
    if isinstance(call, dict):
        name = call.get("name") or call.get("tool_name") or call.get("function_name") or call.get("api_name")
        if not name:
            return None
        args = call.get("arguments") or call.get("args") or call.get("parameters") or call.get("params")
        if args is None:
            args = call.get("input") or call.get("tool_input") or call.get("api_arguments")
        args = _normalize_arguments(args)
        return {"name": str(name), "arguments": args}
    return None


def _coerce_calls(raw: Any) -> Optional[List[dict]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text in ("", "none", "no", "null"):
            return []
        parsed = _parse_json_maybe(text)
        if parsed is not None:
            raw = parsed
        else:
            return None
    if isinstance(raw, dict):
        if "tool_calls" in raw:
            raw = raw["tool_calls"]
        else:
            call = _normalize_call(raw)
            return [call] if call else None
    if isinstance(raw, list):
        calls = []
        for item in raw:
            call = _normalize_call(item)
            if call:
                calls.append(call)
        return calls
    return None


def _extract_calls(row: dict) -> Optional[List[dict]]:
    needs_tool = _extract_first(row, _NO_TOOL_KEYS)
    if isinstance(needs_tool, bool) and not needs_tool:
        return []
    for key in _CALL_KEYS:
        if key in row and row[key] is not None:
            calls = _coerce_calls(row[key])
            if calls is not None:
                return calls
    name = _extract_first(row, ("tool_name", "api_name", "function_name", "name"))
    args = _extract_first(row, ("arguments", "args", "parameters", "params", "input", "tool_input"))
    if name is not None:
        return [{"name": str(name), "arguments": _normalize_arguments(args)}]
    return None


def augment_tools(tools: List[dict], distractor_tools: List[dict], max_distractors: int) -> List[dict]:
    if not tools:
        tools = []
    if not distractor_tools or max_distractors <= 0:
        return tools
    existing = {t.get("name") for t in tools if isinstance(t, dict)}
    extras = []
    for tool in distractor_tools:
        name = tool.get("name") if isinstance(tool, dict) else None
        if not name or name in existing:
            continue
        extras.append(tool)
        if len(extras) >= max_distractors:
            break
    return tools + extras


def build_tool_prompt_messages(tools: List[dict], query: str, use_system_prompt: bool = True) -> List[dict]:
    tools_json = json.dumps(tools, ensure_ascii=False)
    if use_system_prompt:
        system_prompt = (
            "You have access to the following tools (JSON schema):\n"
            f"{tools_json}\n\n"
            f"{_TOOL_PROMPT_INSTRUCTION}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(query)},
        ]
    user_message = (
        "You have access to the following tools (JSON schema):\n"
        f"{tools_json}\n\n"
        f"User query: {query}\n\n"
        f"{_TOOL_PROMPT_INSTRUCTION}"
    )
    return [{"role": "user", "content": user_message}]


def build_tool_conversation(tools: List[dict], query: str, calls: List[dict], use_system_prompt: bool = True) -> dict:
    messages = build_tool_prompt_messages(tools, query, use_system_prompt=use_system_prompt)
    messages.append({"role": "assistant", "content": json.dumps(calls, ensure_ascii=False)})
    return {"messages": messages, "tool_schema": tools}


def _load_dataset_with_fallback(name_candidates: Iterable[Any], split: str, subset: Optional[str]) -> Tuple[Any, str, str]:
    errors = []
    if split == "test":
        split_candidates = ["test", "validation", "val", "dev", "train"]
    elif split == "train":
        split_candidates = ["train", "validation", "val", "test"]
    else:
        split_candidates = [split, "train"]
    for candidate in name_candidates:
        if isinstance(candidate, tuple):
            name, subset_override = candidate
        else:
            name, subset_override = candidate, subset
        for split_name in split_candidates:
            try:
                if subset_override is None:
                    ds = load_dataset(name, split=split_name)
                else:
                    ds = load_dataset(name, subset_override, split=split_name)
                return ds.shuffle(seed=42), split_name, name
            except Exception as exc:
                errors.append(f"{name}:{split_name} -> {exc}")
                continue
    raise ValueError("Failed to load dataset with any fallback split:\n" + "\n".join(errors))


class ToolCallingHFDataset(Task):
    def __init__(
        self,
        name_candidates: Iterable[Any],
        split: str = "train",
        subset: Optional[str] = None,
        add_distractors: int = 0,
        distractor_tools: Optional[List[dict]] = None,
        use_system_prompt: bool = True,
        force_no_tool: bool = False,
        english_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name_candidates = list(name_candidates)
        self.split = split
        self.subset = subset
        self.add_distractors = max(0, int(add_distractors))
        self.distractor_tools = list(distractor_tools) if distractor_tools is not None else list(DEFAULT_TOOL_SCHEMA)
        self.use_system_prompt = use_system_prompt
        self.force_no_tool = force_no_tool
        self.english_only = english_only
        self.ds, self.split_used, self.name_used = _load_dataset_with_fallback(self.name_candidates, split, subset)
        if self.stop is not None and self.stop > len(self.ds):
            self.stop = len(self.ds)
        self._verify_min_valid_samples()

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def _verify_min_valid_samples(self, max_checks: int = 50) -> None:
        if len(self.ds) == 0:
            raise ValueError(f"Dataset {self.name_used}:{self.split_used} is empty")
        checked = min(max_checks, len(self.ds))
        for i in range(checked):
            try:
                row = self.ds[i]
                if self.english_only and not _is_english_row(row):
                    continue
                conversation = self._row_to_conversation(row)
                if not isinstance(conversation, dict) or "messages" not in conversation:
                    raise ValueError("row_to_conversation returned invalid conversation")
                return
            except Exception:
                continue
        raise ValueError(f"Dataset {self.name_used}:{self.split_used} appears incompatible with tool-call parsing")

    def _row_to_conversation(self, row: dict) -> dict:
        query = _extract_query(row)
        if query is None or not str(query).strip():
            raise ValueError("Tool-call row missing query")
        tools = _extract_tools(row)
        if not tools:
            raise ValueError("Tool-call row missing tools")
        calls = [] if self.force_no_tool else _extract_calls(row)
        if calls is None:
            raise ValueError("Tool-call row missing calls")
        tools = augment_tools(list(tools), self.distractor_tools, self.add_distractors)
        calls = list(calls) if isinstance(calls, list) else []
        return build_tool_conversation(tools, str(query), calls, use_system_prompt=self.use_system_prompt)

    def get_example(self, index: int) -> dict:
        if len(self.ds) == 0:
            raise ValueError(f"Dataset {self.name_used}:{self.split_used} is empty")
        attempts = 0
        max_attempts = 100
        idx = index % len(self.ds)
        last_exc = None
        while attempts < max_attempts:
            row = self.ds[idx]
            try:
                if self.english_only and not _is_english_row(row):
                    attempts += 1
                    idx = (idx + 1) % len(self.ds)
                    continue
                conversation = self._row_to_conversation(row)
                return conversation
            except Exception as exc:
                last_exc = exc
                attempts += 1
                idx = (idx + 1) % len(self.ds)
                continue
        raise ValueError(
            f"Failed to parse tool-call row after {max_attempts} attempts "
            f"({self.name_used}:{self.split_used})"
        ) from last_exc


class APIGenToolCalls(ToolCallingHFDataset):
    def __init__(self, split: str = "train", **kwargs):
        name_candidates = [
            "Gorilla-LLM/APIGen",
            "Salesforce/APIGen",
        ]
        super().__init__(name_candidates=name_candidates, split=split, **kwargs)


class ToolBenchToolCalls(ToolCallingHFDataset):
    def __init__(self, split: str = "train", **kwargs):
        name_candidates = [
            "OpenBMB/ToolBench",
            "openbmb/ToolBench",
        ]
        super().__init__(name_candidates=name_candidates, split=split, **kwargs)


class ToolAlpacaToolCalls(ToolCallingHFDataset):
    def __init__(self, split: str = "train", **kwargs):
        name_candidates = [
            "tangqiaoyu/ToolAlpaca",
            "tangqiaoyu/toolalpaca",
        ]
        super().__init__(name_candidates=name_candidates, split=split, **kwargs)


class APIBankToolCalls(ToolCallingHFDataset):
    def __init__(self, split: str = "train", **kwargs):
        name_candidates = [
            "TencentARC/API-Bank",
            "TencentARC/APIBank",
        ]
        super().__init__(name_candidates=name_candidates, split=split, **kwargs)


class APIPackToolCalls(ToolCallingHFDataset):
    def __init__(self, split: str = "train", **kwargs):
        name_candidates = [
            "HuggingFaceH4/APIPack",
            "HuggingFaceH4/apipack",
            "APIPack/APIPack",
        ]
        super().__init__(name_candidates=name_candidates, split=split, **kwargs)


class NoToolTask(Task):
    """Wrap a base Task to create BFCL-style no-tool examples (assistant outputs [])."""
    def __init__(self, base_task: Task, tools: Optional[List[dict]] = None, use_system_prompt: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.base_task = base_task
        self.tools = list(tools) if tools is not None else list(DEFAULT_TOOL_SCHEMA)
        self.use_system_prompt = use_system_prompt

    @property
    def eval_type(self):
        return self.base_task.eval_type

    def num_examples(self):
        return len(self.base_task)

    def _extract_query(self, conversation: dict) -> Optional[str]:
        messages = conversation.get("messages") or []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None

    def get_example(self, index: int) -> dict:
        conversation = self.base_task[index]
        query = self._extract_query(conversation)
        if query is None:
            raise ValueError("NoToolTask: base conversation missing user message")
        return build_tool_conversation(self.tools, query, [], use_system_prompt=self.use_system_prompt)
