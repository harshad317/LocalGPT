"""
Dataset wrappers and helpers for SDPO training across multiple HF datasets.
"""

import ast
import json
import re
import urllib.request
from typing import Callable, Iterable, Optional

from datasets import load_dataset

from .common import Task, render_mc
from .gsm8k import extract_answer
from .hendrycks_math import HendrycksMath
from .alpaca import Alpaca
from .hf_utils import convert_conversations, SYSTEM_ROLES, USER_ROLES, ASSISTANT_ROLES, TOOL_ROLES


def _as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content)
    return str(content)


def _is_english_row(row):
    if not isinstance(row, dict):
        return True
    lang_keys = ("lang", "language", "languages", "langs", "locale", "language_code")
    values = []
    for key in lang_keys:
        if key in row and row[key] is not None:
            values.append(row[key])
    if not values:
        return True
    for val in values:
        if isinstance(val, (list, tuple)):
            tokens = [str(v).lower() for v in val]
        else:
            tokens = [str(val).lower()]
        for token in tokens:
            parts = re.split(r"[\s,;/|]+", token)
            for part in parts:
                if not part:
                    continue
                if part == "eng" or part.startswith("en") or "english" in part:
                    return True
    return False


def _normalize_role(role_raw, is_first):
    role = str(role_raw).strip().lower()
    if role in SYSTEM_ROLES:
        return "system" if is_first else "user"
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    if role in TOOL_ROLES:
        return "user"
    return "user"


def _normalize_messages(messages):
    if not messages:
        raise ValueError("Conversation must have at least one message")
    first = messages[0]
    if isinstance(first, dict) and "from" in first and "value" in first:
        return convert_conversations(messages, role_key="from", content_key="value")
    if isinstance(first, dict) and "role" in first and "content" in first:
        normalized = []
        for i, msg in enumerate(messages):
            role = _normalize_role(msg.get("role"), is_first=i == 0)
            normalized.append({"role": role, "content": _as_text(msg.get("content", ""))})
        return normalized
    raise ValueError("Unsupported message format in conversation")


def _validate_messages(messages):
    assert messages, "Conversation must have at least one message"
    start = 1 if messages[0]["role"] == "system" else 0
    for i, msg in enumerate(messages[start:], start=0):
        expected = "user" if i % 2 == 0 else "assistant"
        assert msg["role"] == expected, f"Message {i} has role {msg['role']} but should be {expected}"
        assert isinstance(msg["content"], str), "Message content must be a string"
    assert messages[-1]["role"] == "assistant", "Conversation must end with an assistant message"


def _message_pair(user_text, assistant_text):
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }

def _normalize_mc_answer(text):
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    match = re.match(r"^[^A-Za-z0-9]*([A-Za-z])[^A-Za-z0-9]*$", text)
    return match.group(1).upper() if match else None


_TEX_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
_TEX_FRAME_RE = re.compile(r"\\framebox\{([^}]*)\}")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_ZERO_WIDTH_LITERAL_RE = re.compile(r"u200b|u200c|u200d|u2060|ufeff", re.IGNORECASE)
_LOGIQA_CONTEXT_RE = re.compile(r"<!--retrieval_context-->(.*?)<!--/retrieval_context-->", re.DOTALL | re.IGNORECASE)
_LOGIQA_INPUT_RE = re.compile(r"<!--input-->(.*?)<!--/input-->", re.DOTALL | re.IGNORECASE)


def _normalize_answer_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    text = text.replace("$", "")
    text = text.replace("\u00a0", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _ZERO_WIDTH_LITERAL_RE.sub("", text)
    text = _TEX_BOX_RE.sub(r"\1", text)
    text = _TEX_FRAME_RE.sub(r"\1", text)
    return re.sub(r"\s+", "", text)


def _extract_mc_options(row):
    options_raw = _extract_first(row, ["options", "choices", "answers", "answer_choices", "choices_text"])
    if options_raw is None:
        return None, None
    if isinstance(options_raw, dict):
        texts = options_raw.get("text") or options_raw.get("options") or options_raw.get("choices")
        labels = options_raw.get("label") or options_raw.get("labels")
        if texts:
            return [str(t) for t in texts], [str(l) for l in labels] if labels else None
    if isinstance(options_raw, list):
        if options_raw and isinstance(options_raw[0], dict):
            texts = []
            labels = []
            for opt in options_raw:
                texts.append(str(opt.get("text") or opt.get("content") or opt.get("value") or ""))
                if "label" in opt:
                    labels.append(str(opt.get("label")))
            return texts, labels if labels else None
        return [str(opt) for opt in options_raw], None
    if isinstance(options_raw, str):
        options = _parse_math_qa_options(options_raw)
        if options:
            return options, None
    return None, None


def _resolve_mc_answer(answer_raw, labels, options):
    if answer_raw is None:
        return None
    if isinstance(answer_raw, (int, float)) and int(answer_raw) == answer_raw:
        idx = int(answer_raw)
        if 0 <= idx < len(options):
            return labels[idx]
    ans = str(answer_raw).strip()
    if not ans:
        return None
    if ans.isdigit():
        idx = int(ans)
        if 0 <= idx < len(options):
            return labels[idx]
    letter = _normalize_mc_answer(ans)
    if letter and letter in labels:
        return letter
    for i, opt in enumerate(options):
        if str(opt).strip() == ans:
            return labels[i]
    ans_norm = _normalize_answer_text(ans).lower()
    if ans_norm:
        for i, opt in enumerate(options):
            if _normalize_answer_text(opt).lower() == ans_norm:
                return labels[i]
    return None


def _build_mc_conversation(question, options, answer_raw=None, labels=None):
    if labels:
        labels = [str(l).strip().upper() for l in labels]
    else:
        labels = [chr(ord("A") + i) for i in range(len(options))]
    answer_letter = _resolve_mc_answer(answer_raw, labels, options)
    if answer_letter is None:
        raise ValueError("Multiple-choice row missing answer")
    user_message = render_mc(str(question), labels, [str(opt) for opt in options])
    conversation = _message_pair(user_message, answer_letter)
    conversation["letters"] = labels
    return conversation


def _mc_letter_eval(conversation, assistant_response):
    if not isinstance(assistant_response, str):
        return False
    correct_letter = _normalize_mc_answer(conversation["messages"][-1]["content"])
    if not correct_letter:
        return False
    resp = assistant_response.strip()
    if not resp:
        return False
    match = re.search(r"\b([A-Za-z])\b", resp)
    if match and match.group(1).upper() == correct_letter:
        return True
    if resp.strip().upper().startswith(correct_letter):
        return True
    return False

def _extract_first(row, keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_hh_conversation(text):
    chunks = re.split(r"\n\n(?=Human:|Assistant:)", text.strip())
    messages = []
    for chunk in chunks:
        if chunk.startswith("Human:"):
            messages.append({"role": "user", "content": chunk[len("Human:"):].strip()})
        elif chunk.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": chunk[len("Assistant:"):].strip()})
    if not messages:
        raise ValueError("Empty HH conversation")
    if messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    _validate_messages(messages)
    return messages


def _parse_math_qa_options(options_raw):
    if isinstance(options_raw, list):
        return options_raw
    if not isinstance(options_raw, str):
        return []
    # Try to parse "a) ... b) ... c) ..." style options.
    matches = re.findall(r"([a-e])\s*\)\s*([^,]+)", options_raw, flags=re.IGNORECASE)
    if matches:
        return [text.strip() for _, text in matches]
    # Fallback: split on " , "
    return [opt.strip() for opt in options_raw.split(",") if opt.strip()]


def _parse_bracketed_options(text):
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    quoted_items = []
    current = []
    quote = None
    escape = False
    for ch in text:
        if quote:
            if escape:
                current.append(ch)
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quoted_items.append("".join(current))
                current = []
                quote = None
            else:
                current.append(ch)
        else:
            if ch in ("'", "\""):
                quote = ch
    if len(quoted_items) >= 2:
        return [item.strip() for item in quoted_items if item and item.strip()]
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(opt) for opt in parsed if opt not in (None, "")]
    if quoted_items:
        return [item.strip() for item in quoted_items if item and item.strip()]
    cleaned = text.strip().strip("[]")
    parts = [p.strip().strip("\"'") for p in re.split(r"\s*\n\s*|\\n|\\t|\\r|\\s{2,}|\\s*;\\s*", cleaned)]
    return [p for p in parts if p]


def _numeric_match(a, b):
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except Exception:
        return str(a).strip() == str(b).strip()


class HFDatasetTask(Task):
    def __init__(
        self,
        name_candidates: Iterable[str],
        split: str,
        row_to_conversation: Callable,
        eval_fn: Optional[Callable] = None,
        subset: Optional[str] = None,
        english_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name_candidates = list(name_candidates)
        self.row_to_conversation = row_to_conversation
        self.eval_fn = eval_fn
        self.split = split
        self.subset = subset
        self.english_only = english_only
        self.ds, self.split_used, self.name_used = _load_dataset_with_fallback(self.name_candidates, split, subset)
        self._verify_min_valid_samples()

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def _verify_min_valid_samples(self, max_checks=50):
        if len(self.ds) == 0:
            raise ValueError(f"Dataset {self.name_used}:{self.split_used} is empty")
        checked = min(max_checks, len(self.ds))
        for i in range(checked):
            try:
                row = self.ds[i]
                if self.english_only and not _is_english_row(row):
                    continue
                conversation = self.row_to_conversation(row)
                if not isinstance(conversation, dict) or "messages" not in conversation:
                    raise ValueError("row_to_conversation returned invalid conversation")
                return
            except Exception:
                continue
        raise ValueError(
            f"Dataset {self.name_used}:{self.split_used} appears incompatible with row_to_conversation"
        )

    def get_example(self, index):
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
                conversation = self.row_to_conversation(row)
                if not isinstance(conversation, dict) or "messages" not in conversation:
                    raise ValueError("row_to_conversation returned invalid conversation")
                return conversation
            except Exception as exc:
                last_exc = exc
                attempts += 1
                idx = (idx + 1) % len(self.ds)
                continue
        raise ValueError(
            f"Failed to parse dataset row after {max_attempts} attempts "
            f"({self.name_used}:{self.split_used})"
        ) from last_exc

    def evaluate(self, conversation, assistant_response):
        if self.eval_fn is None:
            if not isinstance(assistant_response, str):
                return False
            ref = conversation["messages"][-1]["content"]
            return assistant_response.strip() == _as_text(ref).strip()
        return self.eval_fn(conversation, assistant_response)


def _load_dataset_with_fallback(name_candidates, split, subset):
    errors = []
    repo_files_only = {
        "allenai/math_qa",
        "lucasmccabe/logiqa",
        "allenai/social_i_qa",
    }
    if split == "test":
        split_candidates = ["test", "validation", "val", "dev", "train"]
    elif split == "train":
        split_candidates = ["train", "validation", "val", "test"]
    else:
        split_candidates = [split, "train"]
    for name in name_candidates:
        use_repo_files = str(name).lower() in repo_files_only
        for split_name in split_candidates:
            try:
                if use_repo_files:
                    ds = _load_repo_files_dataset(name, split_name, subset)
                else:
                    if subset is None:
                        ds = load_dataset(name, split=split_name)
                    else:
                        ds = load_dataset(name, subset, split=split_name)
                return ds.shuffle(seed=42), split_name, name
            except Exception as e:
                errors.append(f"{name}:{split_name} -> {e}")
                continue
    raise ValueError("Failed to load dataset with any fallback split:\n" + "\n".join(errors))


_PARQUET_INDEX_CACHE = {}
_REPO_FILE_CACHE = {}
_DATA_FILE_EXTS = (".parquet", ".json", ".jsonl", ".csv", ".tsv")
_SPLIT_TOKENS = {
    "train": ["train"],
    "validation": ["validation", "valid", "val", "dev"],
    "val": ["val", "valid", "validation", "dev"],
    "dev": ["dev", "validation", "valid", "val"],
    "test": ["test"],
}


def _fetch_parquet_index(dataset_name: str):
    if dataset_name in _PARQUET_INDEX_CACHE:
        return _PARQUET_INDEX_CACHE[dataset_name]
    api_urls = [
        f"https://huggingface.co/api/datasets/{dataset_name}/parquet",
        f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}",
    ]
    last_exc = None
    payload = None
    for url in api_urls:
        try:
            with urllib.request.urlopen(url) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last_exc = exc
            continue
    if payload is None:
        raise ValueError(f"Failed to fetch parquet index for {dataset_name}: {last_exc}")
    files = payload.get("parquet_files")
    if files is None and isinstance(payload, list):
        files = payload
    if files is None and isinstance(payload, dict):
        files = payload.get("files") or payload.get("data", {}).get("parquet_files")
    files = files or []
    _PARQUET_INDEX_CACHE[dataset_name] = files
    return files


def _select_parquet_config(files, subset: Optional[str]) -> Optional[str]:
    configs = {f.get("config") for f in files if f.get("config")}
    if subset:
        return subset if subset in configs else None
    if not configs:
        return None
    if "default" in configs:
        return "default"
    if len(configs) == 1:
        return next(iter(configs))
    return sorted(configs)[0]


def _parquet_urls_for_split(files, config: str, split_name: str):
    urls = [
        f.get("url")
        for f in files
        if f.get("config") == config and f.get("split") == split_name and f.get("url")
    ]
    if urls:
        return urls
    partial_split = f"partial-{split_name}"
    return [
        f.get("url")
        for f in files
        if f.get("config") == config and f.get("split") == partial_split and f.get("url")
    ]


def _load_parquet_dataset(dataset_name: str, split_name: str, subset: Optional[str]):
    files = _fetch_parquet_index(dataset_name)
    if not files:
        raise ValueError(f"No parquet files found for {dataset_name}")
    config = _select_parquet_config(files, subset)
    if config is None:
        raise ValueError(f"No parquet config matches subset '{subset}' for {dataset_name}")
    urls = _parquet_urls_for_split(files, config, split_name)
    if not urls:
        raise ValueError(f"No parquet files for split '{split_name}' ({config}) in {dataset_name}")
    return load_dataset("parquet", data_files={split_name: urls}, split=split_name)


def _list_repo_files(dataset_name: str):
    if dataset_name in _REPO_FILE_CACHE:
        return _REPO_FILE_CACHE[dataset_name]
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to list dataset files") from exc
    api = HfApi()
    files = api.list_repo_files(dataset_name, repo_type="dataset")
    _REPO_FILE_CACHE[dataset_name] = files
    return files


def _filter_repo_files(files, subset: Optional[str]):
    data_files = [f for f in files if f.lower().endswith(_DATA_FILE_EXTS)]
    if not subset:
        return data_files
    prefix = f"{subset}/"
    subset_files = [f for f in data_files if f.startswith(prefix)]
    if subset_files:
        return subset_files
    subset_files = [f for f in data_files if f"/{subset}/" in f]
    return subset_files if subset_files else data_files


def _split_matches(name: str, split_name: str) -> bool:
    tokens = _SPLIT_TOKENS.get(split_name, [split_name])
    lower = name.lower()
    return any(token in lower for token in tokens)


def _has_any_split_tokens(files):
    all_tokens = {"train", "validation", "valid", "val", "dev", "test"}
    return any(any(token in f.lower() for token in all_tokens) for f in files)


def _pick_files_for_split(files, split_name: str):
    matched = [f for f in files if _split_matches(f, split_name)]
    if matched:
        return matched
    if split_name == "train" and not _has_any_split_tokens(files):
        return list(files)
    return []


def _choose_extension(files):
    counts = {}
    for name in files:
        lower = name.lower()
        for ext in _DATA_FILE_EXTS:
            if lower.endswith(ext):
                counts[ext] = counts.get(ext, 0) + 1
                break
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _download_repo_files(dataset_name: str, files):
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download dataset files") from exc
    local_paths = []
    for filename in files:
        local_paths.append(hf_hub_download(dataset_name, filename, repo_type="dataset"))
    return local_paths


def _load_repo_files_dataset(dataset_name: str, split_name: str, subset: Optional[str]):
    files = _list_repo_files(dataset_name)
    files = _filter_repo_files(files, subset)
    if not files:
        raise ValueError(f"No data files found in dataset repo {dataset_name}")
    split_files = _pick_files_for_split(files, split_name)
    if not split_files:
        raise ValueError(f"No files for split '{split_name}' in dataset repo {dataset_name}")
    ext = _choose_extension(split_files)
    if ext is None:
        raise ValueError(f"Unsupported data files in dataset repo {dataset_name}")
    split_files = [f for f in split_files if f.lower().endswith(ext)]
    local_paths = _download_repo_files(dataset_name, split_files)
    data_files = {split_name: local_paths}
    if ext in (".json", ".jsonl"):
        return load_dataset("json", data_files=data_files, split=split_name)
    if ext == ".parquet":
        return load_dataset("parquet", data_files=data_files, split=split_name)
    if ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        return load_dataset("csv", data_files=data_files, split=split_name, delimiter=delimiter)
    raise ValueError(f"Unsupported data extension '{ext}' for dataset {dataset_name}")


def _gsm8k_like_row(row):
    question = _extract_first(row, ["question", "Question", "prompt", "problem", "query"])
    answer = _extract_first(row, ["answer", "Answer", "solution", "output", "response"])
    if question is None or answer is None:
        raise ValueError("GSM8K-like row missing question/answer")
    return _message_pair(str(question), str(answer))


def _gsm8k_like_eval(conversation, assistant_response):
    if not isinstance(assistant_response, str):
        return False
    ref = conversation["messages"][-1]["content"]
    ref_num = extract_answer(_as_text(ref))
    pred_num = extract_answer(assistant_response)
    return ref_num is not None and pred_num is not None and ref_num == pred_num


def _calc_svamp_row(row):
    body = _extract_first(row, ["Body", "body", "context"])
    question = _extract_first(row, ["Question", "question", "query", "prompt", "problem"])
    answer = _extract_first(row, ["Answer", "answer", "label", "result", "result_float"])
    if question is None:
        raise ValueError("SVAMP row missing question")
    if answer is None:
        chain = _extract_first(row, ["chain", "solution", "rationale"])
        if isinstance(chain, str):
            match = re.search(r"<result>([^<]+)</result>", chain)
            if match:
                answer = match.group(1).strip()
    if answer is None:
        raise ValueError("SVAMP row missing answer")
    user_message = f"{body.strip()} {question.strip()}" if body else str(question)
    return _message_pair(user_message, str(answer))


def _calc_svamp_eval(conversation, assistant_response):
    if not isinstance(assistant_response, str):
        return False
    ref = conversation["messages"][-1]["content"]
    num_re = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
    def _extract_num(text):
        if text is None:
            return None
        matches = num_re.findall(str(text))
        if not matches:
            return None
        return matches[-1].replace(",", "")
    ref_num = extract_answer(str(ref)) or _extract_num(ref)
    pred_num = extract_answer(assistant_response) or _extract_num(assistant_response)
    return _numeric_match(ref_num, pred_num)


def _math_qa_row(row):
    problem = _extract_first(row, ["Problem", "problem", "question", "query", "prompt"])
    options_raw = _extract_first(row, ["options", "Options", "choices"])
    correct = _extract_first(row, ["correct", "Correct", "answer", "label"])
    if problem is None or correct is None:
        raise ValueError("MathQA row missing problem/correct")
    options = _parse_math_qa_options(options_raw) if options_raw is not None else []
    if options:
        letters = [chr(ord("A") + i) for i in range(len(options))]
        user_message = render_mc(str(problem), letters, options)
        conversation = _message_pair(user_message, str(correct).strip().upper())
        conversation["letters"] = letters
        return conversation
    return _message_pair(str(problem), str(correct))


def _math_qa_eval(conversation, assistant_response):
    if not isinstance(assistant_response, str):
        return False
    ref = conversation["messages"][-1]["content"]
    return assistant_response.strip().upper() == str(ref).strip().upper()

def _commonsense_qa_row(row):
    question = _extract_first(row, ["question", "prompt", "query"])
    choices = row.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if question is None or not texts:
        raise ValueError("CommonsenseQA row missing question/choices")
    if not labels:
        labels = [chr(ord("A") + i) for i in range(len(texts))]
    user_message = render_mc(str(question), labels, [str(t) for t in texts])
    answer = _normalize_mc_answer(row.get("answerKey"))
    if answer is None:
        raise ValueError("CommonsenseQA row missing answerKey")
    conversation = _message_pair(user_message, answer)
    conversation["letters"] = labels
    return conversation


def _piqa_row(row):
    goal = _extract_first(row, ["goal", "question", "prompt"])
    sol1 = _extract_first(row, ["sol1", "option1", "choice1"])
    sol2 = _extract_first(row, ["sol2", "option2", "choice2"])
    label = _extract_first(row, ["label", "answer"])
    if goal is None or sol1 is None or sol2 is None or label is None:
        raise ValueError("PIQA row missing fields")
    options = [str(sol1), str(sol2)]
    labels = ["A", "B"]
    user_message = render_mc(str(goal), labels, options)
    answer_letter = labels[int(label)]
    conversation = _message_pair(user_message, answer_letter)
    conversation["letters"] = labels
    return conversation


def _winogrande_row(row):
    sentence = _extract_first(row, ["sentence", "question", "prompt"])
    option1 = _extract_first(row, ["option1", "choice1"])
    option2 = _extract_first(row, ["option2", "choice2"])
    answer = _extract_first(row, ["answer", "label"])
    if sentence is None or option1 is None or option2 is None or answer is None:
        raise ValueError("Winogrande row missing fields")
    labels = ["A", "B"]
    user_message = render_mc(str(sentence), labels, [str(option1), str(option2)])
    answer_letter = labels[int(answer) - 1] if str(answer).strip().isdigit() else _normalize_mc_answer(answer)
    if answer_letter is None:
        raise ValueError("Winogrande row missing answer")
    conversation = _message_pair(user_message, answer_letter)
    conversation["letters"] = labels
    return conversation


def _cosmos_qa_row(row):
    context = _extract_first(row, ["context", "passage", "story"])
    question = _extract_first(row, ["question", "query", "prompt"])
    label = _extract_first(row, ["label", "answer"])
    answers = [
        _extract_first(row, ["answer0", "option0", "choice0"]),
        _extract_first(row, ["answer1", "option1", "choice1"]),
        _extract_first(row, ["answer2", "option2", "choice2"]),
        _extract_first(row, ["answer3", "option3", "choice3"]),
    ]
    if context is None or question is None or label is None or any(a is None for a in answers):
        raise ValueError("CosmosQA row missing fields")
    labels = ["A", "B", "C", "D"]
    user_message = render_mc(f"{context}\n\nQuestion: {question}", labels, [str(a) for a in answers])
    answer_letter = labels[int(label)]
    conversation = _message_pair(user_message, answer_letter)
    conversation["letters"] = labels
    return conversation


def _boolq_row(row):
    passage = _extract_first(row, ["passage", "context"])
    question = _extract_first(row, ["question", "prompt"])
    answer = _extract_first(row, ["answer", "label"])
    if passage is None or question is None or answer is None:
        raise ValueError("BoolQ row missing fields")
    answer_val = answer
    if isinstance(answer, str):
        ans = answer.strip().lower()
        if ans in ("true", "yes", "1"):
            answer_val = True
        elif ans in ("false", "no", "0"):
            answer_val = False
    user_message = f"{passage}\n\nQuestion: {question}\n\nAnswer yes or no."
    target = "yes" if bool(answer_val) else "no"
    return _message_pair(user_message, target)


def _boolq_eval(conversation, assistant_response):
    if not isinstance(assistant_response, str):
        return False
    ref = conversation["messages"][-1]["content"].strip().lower()
    resp = assistant_response.strip().lower()
    if not resp:
        return False
    if ref in ("yes", "no") and resp.startswith(ref):
        return True
    if ref == "yes" and any(tok in resp for tok in ["true", "yes"]):
        return True
    if ref == "no" and any(tok in resp for tok in ["false", "no"]):
        return True
    return False


def _race_row(row):
    article = _extract_first(row, ["article", "passage", "context"])
    question = _extract_first(row, ["question", "query", "prompt"])
    options = _extract_first(row, ["options", "choices"])
    answer = _extract_first(row, ["answer", "label"])
    if article is None or question is None or options is None or answer is None:
        raise ValueError("RACE row missing fields")
    labels = [chr(ord("A") + i) for i in range(len(options))]
    user_message = render_mc(f"{article}\n\nQuestion: {question}", labels, [str(o) for o in options])
    answer_letter = _normalize_mc_answer(answer)
    if answer_letter is None:
        raise ValueError("RACE row missing answer")
    conversation = _message_pair(user_message, answer_letter)
    conversation["letters"] = labels
    return conversation


def _social_iqa_row(row):
    context = _extract_first(row, ["context", "story", "passage"])
    question = _extract_first(row, ["question", "query", "prompt"])
    if question is None:
        raise ValueError("SocialIQA row missing question")
    options = []
    for key in ("answerA", "answerB", "answerC", "answer_a", "answer_b", "answer_c", "optionA", "optionB", "optionC"):
        val = row.get(key)
        if val not in (None, ""):
            options.append(val)
    if len(options) < 2:
        opts, _labels = _extract_mc_options(row)
        if opts:
            options = opts
    if len(options) < 2:
        raise ValueError("SocialIQA row missing options")
    answer_raw = _extract_first(row, ["label", "answer", "correct", "answerKey"])
    if answer_raw is None:
        raise ValueError("SocialIQA row missing label")
    try:
        label_int = int(str(answer_raw).strip())
        if 1 <= label_int <= len(options):
            answer_raw = label_int - 1
        elif 0 <= label_int < len(options):
            answer_raw = label_int
    except Exception:
        pass
    prompt = f"{context}\n\nQuestion: {question}" if context else str(question)
    return _build_mc_conversation(prompt, options, answer_raw=answer_raw)


def _qasc_row(row):
    question = _extract_first(row, ["question", "formatted_question"])
    if question is None:
        raise ValueError("QASC row missing question")
    options, labels = _extract_mc_options(row)
    if not options:
        raise ValueError("QASC row missing options")
    answer_raw = _extract_first(row, ["answerKey", "answer", "label", "correct"])
    if answer_raw is None:
        raise ValueError("QASC row missing answerKey")
    return _build_mc_conversation(question, options, answer_raw=answer_raw, labels=labels)


def _parse_logiqa_input(text):
    if text is None:
        return None, None, None
    text = str(text)
    context = None
    match = _LOGIQA_CONTEXT_RE.search(text)
    if match:
        context = match.group(1).strip()
    input_block = text
    match = _LOGIQA_INPUT_RE.search(text)
    if match:
        input_block = match.group(1).strip()
    input_block = input_block.strip()
    question = input_block
    options = None
    start = input_block.find("[")
    end = input_block.rfind("]")
    if start != -1 and end != -1 and end > start:
        list_text = input_block[start:end + 1]
        options = _parse_bracketed_options(list_text)
        question = input_block[:start].strip()
    if question:
        question = question.strip().lstrip(",")
        question = question.strip()
    return context, question or None, options


def _logiqa_row(row):
    context = _extract_first(row, ["context", "passage", "text"])
    question = _extract_first(row, ["query", "question", "prompt"])
    options, labels = _extract_mc_options(row)
    answer_raw = _extract_first(row, ["correct_option", "answer", "label", "answer_index"])

    if question is None or not options:
        input_text = _extract_first(row, ["input", "instruction", "prompt"])
        parsed_context, parsed_question, parsed_options = _parse_logiqa_input(input_text)
        if context is None:
            context = parsed_context
        if question is None:
            question = parsed_question
        if not options and parsed_options:
            options = parsed_options
    if answer_raw is None:
        answer_raw = _extract_first(row, ["expected_output", "output", "completion"])

    if question is None:
        raise ValueError("LogiQA row missing question")
    if not options:
        raise ValueError("LogiQA row missing options")
    if answer_raw is None:
        raise ValueError("LogiQA row missing answer")
    prompt = f"{context}\n\nQuestion: {question}" if context else str(question)
    return _build_mc_conversation(prompt, options, answer_raw=answer_raw, labels=labels)


def _aqua_rat_mcqa_row(row):
    question = _extract_first(row, ["question", "problem", "prompt"])
    if question is None:
        raise ValueError("AQuA-RAT row missing question")
    options, labels = _extract_mc_options(row)
    if not options:
        choices = row.get("choices") or row.get("options")
        if isinstance(choices, list):
            options = choices
            labels = None
    if not options:
        raise ValueError("AQuA-RAT row missing options")
    answer_raw = _extract_first(row, ["answer_index", "correct", "label", "answer", "answerKey", "answer_text"])
    if answer_raw is None:
        raise ValueError("AQuA-RAT row missing answer")
    return _build_mc_conversation(question, options, answer_raw=answer_raw, labels=labels)


def _instruction_row(row):
    instruction = _extract_first(row, ["instruction", "prompt", "question", "query"])
    context = _extract_first(row, ["context", "input"])
    response = _extract_first(row, ["output", "response", "answer", "completion"])
    if instruction is None or response is None:
        raise ValueError("Instruction row missing instruction/response")
    if context and str(context).strip():
        user_message = f"{instruction}\n\nContext:\n{context}"
    else:
        user_message = str(instruction)
    return _message_pair(user_message, str(response))


def _prompt_completion_row(row):
    prompt = _extract_first(row, ["prompt", "input", "question", "instruction", "problem", "Problem", "query", "task", "text"])
    completion = _extract_first(row, ["completion", "output", "response", "answer", "solution", "Solution", "final", "final_answer", "target", "label", "rationale", "explanation"])
    if prompt is None or completion is None:
        raise ValueError("Prompt/completion row missing fields")
    return _message_pair(str(prompt), str(completion))


def _flexible_answer_eval(conversation, assistant_response):
    if not isinstance(assistant_response, str):
        return False
    ref = conversation.get("answer")
    if ref is None:
        ref = conversation["messages"][-1]["content"]
    ref_text = _as_text(ref)
    ref_num = extract_answer(ref_text)
    pred_num = extract_answer(assistant_response)
    if ref_num is not None and pred_num is not None:
        return _numeric_match(ref_num, pred_num)
    return _normalize_answer_text(ref_text).lower() == _normalize_answer_text(assistant_response).lower()


def _mixed_math_eval(conversation, assistant_response):
    if isinstance(conversation, dict) and conversation.get("letters"):
        return _mc_letter_eval(conversation, assistant_response)
    return _flexible_answer_eval(conversation, assistant_response)


def _verifiable_corpus_row(row):
    try:
        return _messages_row(row)
    except Exception:
        pass
    prompt = _extract_first(row, ["problem", "question", "prompt", "input", "instruction", "description"])
    answer = _extract_first(row, ["answer", "response", "output", "completion", "solution"])
    if prompt is None or answer is None:
        raise ValueError("verifiable-corpus row missing prompt/answer")
    description = _extract_first(row, ["description", "context"])
    if description and str(description).strip() and str(description).strip() not in str(prompt):
        prompt = f"{description}\n\n{prompt}"
    return _message_pair(str(prompt), str(answer))


def _gpqa_d_row(row):
    question = _extract_first(row, ["Question", "question", "prompt", "problem", "query"])
    if question is None:
        raise ValueError("GPQA-D row missing question")
    options, labels = _extract_mc_options(row)
    if options:
        answer_raw = _extract_first(row, ["answer_index", "answer", "label", "correct"])
        return _build_mc_conversation(question, options, answer_raw=answer_raw, labels=labels)
    correct = _extract_first(row, ["Correct Answer", "correct_answer", "correct", "answer"])
    incorrect = []
    for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Incorrect Answer 4"):
        val = row.get(key)
        if val not in (None, ""):
            incorrect.append(val)
    alt_incorrect = row.get("incorrect_answers") or row.get("distractors") or row.get("wrong_answers")
    if isinstance(alt_incorrect, list):
        incorrect.extend([v for v in alt_incorrect if v not in (None, "")])
    if correct is None or not incorrect:
        raise ValueError("GPQA-D row missing answers")
    options = [correct] + incorrect
    return _build_mc_conversation(question, options, answer_raw=0)


def _mmlu_pro_row(row):
    question = _extract_first(row, ["question", "prompt", "problem", "query"])
    if question is None:
        raise ValueError("MMLU-Pro row missing question")
    options, labels = _extract_mc_options(row)
    if not options:
        raise ValueError("MMLU-Pro row missing options")
    answer_raw = _extract_first(row, ["answer_index", "answer", "label"])
    return _build_mc_conversation(question, options, answer_raw=answer_raw, labels=labels)


def _math_ai_row(row):
    question = _extract_first(row, ["problem", "question", "prompt", "query", "instruction"])
    if question is None:
        raise ValueError("math-ai row missing question")
    options, labels = _extract_mc_options(row)
    if options:
        answer_raw = _extract_first(row, ["answer", "label", "correct", "answer_index"])
        return _build_mc_conversation(question, options, answer_raw=answer_raw, labels=labels)
    solution = _extract_first(row, ["solution", "rationale", "explanation"])
    answer = _extract_first(row, ["answer", "final_answer", "output", "response", "label"])
    if solution is None and answer is None:
        raise ValueError("math-ai row missing solution/answer")
    assistant = solution if solution is not None and str(solution).strip() else answer
    conversation = _message_pair(str(question), str(assistant))
    if answer is not None:
        conversation["answer"] = answer
    return conversation

def _mbpp_row(row):
    prompt = _extract_first(row, ["text", "prompt", "question", "instruction"])
    code = _extract_first(row, ["code", "completion", "output", "response", "answer"])
    if prompt is None or code is None:
        raise ValueError("MBPP row missing text/code")
    return _message_pair(str(prompt), str(code))


def _open_orca_row(row):
    system_prompt = _extract_first(row, ["system_prompt", "system"])
    question = _extract_first(row, ["question", "prompt", "instruction", "query"])
    response = _extract_first(row, ["response", "output", "answer", "completion"])
    if question is None or response is None:
        raise ValueError("OpenOrca row missing question/response")
    messages = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(question)})
    messages.append({"role": "assistant", "content": str(response)})
    return {"messages": messages}


def _messages_row(row):
    raw = _extract_first(row, ["messages", "conversation", "conversations"])
    if raw is None:
        raise ValueError("Row missing messages list")
    messages = _normalize_messages(raw)
    if messages and messages[-1]["role"] != "assistant":
        messages = messages[:-1]
    _validate_messages(messages)
    return {"messages": messages}


class OASST1Pairs(Task):
    def __init__(self, name_candidates, split="train", **kwargs):
        super().__init__(**kwargs)
        self.ds, self.split_used, self.name_used = _load_dataset_with_fallback(name_candidates, split, subset=None)
        by_id = {}
        for row in self.ds:
            msg_id = row.get("message_id")
            if msg_id:
                by_id[msg_id] = row
        pairs = []
        for row in self.ds:
            role = str(row.get("role", "")).lower()
            if role != "assistant":
                continue
            parent_id = row.get("parent_id")
            if not parent_id or parent_id not in by_id:
                continue
            parent = by_id[parent_id]
            parent_role = str(parent.get("role", "")).lower()
            if parent_role not in ("prompter", "user", "human"):
                continue
            if row.get("lang") and row.get("lang") != "en":
                continue
            if parent.get("lang") and parent.get("lang") != "en":
                continue
            user_text = parent.get("text")
            assistant_text = row.get("text")
            if not user_text or not assistant_text:
                continue
            pairs.append(_message_pair(user_text, assistant_text))
        self.pairs = pairs

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.pairs)

    def get_example(self, index):
        return self.pairs[index]


def _build_gsm8k(split):
    return [GSM8KLikeTask("openai/gsm8k", split, subset="main")]


def _build_gsm8k_platinum(split):
    return [GSM8KLikeTask("madrylab/gsm8k-platinum", split)]


def _build_gsm8k_567(split):
    return [GSM8KLikeTask("567-labs/gsm8k", split)]


def _build_hendrycks_math(split):
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    return [HendrycksMath(subject, split) for subject in subjects]


def _build_hendrycks_math_benchmark(split):
    return [HFDatasetTask(["nlile/hendrycks-MATH-benchmark"], split, _prompt_completion_row)]


def _build_math_qa(split):
    return [HFDatasetTask(["regisss/math_qa", "allenai/math_qa"], split, _math_qa_row, _math_qa_eval)]


def _build_calc_svamp(split):
    return [HFDatasetTask(["MU-NLPC/Calc-svamp"], split, _calc_svamp_row, _calc_svamp_eval)]


def _build_math_augmented(split):
    return [HFDatasetTask(["nivektk/math-augmented-dataset"], split, _gsm8k_like_row, _gsm8k_like_eval)]


def _build_alpaca(split):
    if split != "train":
        split = "train"
    return [Alpaca(split=split)]


def _build_dolly(split):
    return [HFDatasetTask(["aisquared/databricks-dolly-15k", "databricks/databricks-dolly-15k"], split, _instruction_row)]


def _build_hh_rlhf(split):
    def row_fn(row):
        chosen = _extract_first(row, ["chosen", "response", "text"])
        if chosen is None:
            raise ValueError("HH-RLHF row missing chosen")
        messages = _parse_hh_conversation(chosen)
        return {"messages": messages}
    return [HFDatasetTask(["Anthropic/hh-rlhf"], split, row_fn)]


def _build_lmsys(split):
    return [HFDatasetTask(["lmsys/lmsys-chat-1m"], split, _messages_row)]


def _build_oasst1(split):
    return [OASST1Pairs(["OpenAssistant/oasst1"], split=split)]


def _build_oasst1_h2o(split):
    return [HFDatasetTask(["h2oai/openassistant_oasst1"], split, _instruction_row)]


def _build_json_mermaid(split):
    def row_fn(row):
        json_text = _extract_first(row, ["json", "input", "prompt", "instruction"])
        mermaid_text = _extract_first(row, ["mermaid", "output", "response", "completion"])
        if json_text is None or mermaid_text is None:
            raise ValueError("json-mermaid row missing json/mermaid")
        return _message_pair(str(json_text), str(mermaid_text))
    return [HFDatasetTask(["moogin/json-mermaid", "mugivara1/json-mermaid"], split, row_fn)]

def _build_commonsense_qa(split):
    return [HFDatasetTask(["commonsense_qa"], split, _commonsense_qa_row, _mc_letter_eval)]


def _build_piqa(split):
    return [HFDatasetTask(["piqa"], split, _piqa_row, _mc_letter_eval)]


def _build_winogrande(split):
    return [HFDatasetTask(["winogrande"], split, _winogrande_row, _mc_letter_eval)]


def _build_cosmos_qa(split):
    return [HFDatasetTask(["cosmos_qa"], split, _cosmos_qa_row, _mc_letter_eval)]


def _build_boolq(split):
    return [HFDatasetTask(["boolq"], split, _boolq_row, _boolq_eval)]


def _build_race(split):
    return [
        HFDatasetTask(["race"], split, _race_row, _mc_letter_eval, subset="high"),
        HFDatasetTask(["race"], split, _race_row, _mc_letter_eval, subset="middle"),
    ]


def _build_social_iqa(split):
    return [HFDatasetTask(["jet-ai/social_i_qa", "HappyEval/Social_i_qa-text", "allenai/social_i_qa"], split, _social_iqa_row, _mc_letter_eval)]


def _build_qasc(split):
    return [HFDatasetTask(["allenai/qasc"], split, _qasc_row, _mc_letter_eval)]


def _build_logiqa(split):
    return [HFDatasetTask(["heka-ai/logiqa", "lucasmccabe/logiqa"], split, _logiqa_row, _mc_letter_eval)]


def _build_aqua_rat_mcqa(split):
    return [
        HFDatasetTask(
            ["RikoteMaster/aqua-rat-mcqa", "deepmind/aqua_rat"],
            split,
            _aqua_rat_mcqa_row,
            _mc_letter_eval,
        )
    ]


def _build_mbpp(split):
    return [HFDatasetTask(["google-research-datasets/mbpp"], split, _mbpp_row)]


def _build_code_bagel(split):
    if split != "train":
        split = "train"
    return [HFDatasetTask(["Replete-AI/code_bagel"], split, _instruction_row)]


def _build_z1_code_reasoning(split):
    if split != "train":
        split = "train"
    return [HFDatasetTask(["efficientscaling/Z1-Code-Reasoning-107K"], split, _prompt_completion_row)]


def _build_ultrachat_200k(split):
    split_name = "train_sft" if split == "train" else "test_sft"
    return [HFDatasetTask(["HuggingFaceH4/ultrachat_200k"], split_name, _messages_row)]


def _build_open_orca(split):
    return [HFDatasetTask(["Open-Orca/OpenOrca"], split, _open_orca_row)]


def _build_sharegpt_52k(split):
    return [HFDatasetTask(["RyokoAI/ShareGPT52K"], split, _messages_row)]


def _build_verifiable_corpus(split):
    return [HFDatasetTask(["lasgroup/verifiable-corpus"], split, _verifiable_corpus_row)]


def _build_gpqa_d(split):
    try:
        return [HFDatasetTask(["Idavidrein/gpqa-D"], split, _gpqa_d_row, _mc_letter_eval)]
    except Exception:
        return [HFDatasetTask(["Idavidrein/gpqa"], split, _gpqa_d_row, _mc_letter_eval, subset="gpqa_diamond")]


def _build_mmlu_pro(split):
    return [HFDatasetTask(["TIGER-Lab/MMLU-Pro"], split, _mmlu_pro_row, _mc_letter_eval)]


def _build_aime24(split):
    return [HFDatasetTask(["math-ai/aime24"], split, _math_ai_row, _mixed_math_eval)]


def _build_aime25(split):
    return [HFDatasetTask(["math-ai/aime25"], split, _math_ai_row, _mixed_math_eval)]


def _build_math500(split):
    return [HFDatasetTask(["math-ai/math500"], split, _math_ai_row, _mixed_math_eval)]


def _build_amc23(split):
    return [HFDatasetTask(["math-ai/amc23"], split, _math_ai_row, _mixed_math_eval)]


DATASET_BUILDERS = {
    "gsm8k": _build_gsm8k,
    "gsm8k-platinum": _build_gsm8k_platinum,
    "gsm8k-567": _build_gsm8k_567,
    "hendrycks-math": _build_hendrycks_math,
    "hendrycks-math-benchmark": _build_hendrycks_math_benchmark,
    "math_qa": _build_math_qa,
    "calc-svamp": _build_calc_svamp,
    "math-augmented": _build_math_augmented,
    "alpaca": _build_alpaca,
    "dolly-15k": _build_dolly,
    "hh-rlhf": _build_hh_rlhf,
    "lmsys-chat-1m": _build_lmsys,
    "oasst1": _build_oasst1,
    "oasst1-h2oai": _build_oasst1_h2o,
    "json-mermaid": _build_json_mermaid,
    "ultrachat-200k": _build_ultrachat_200k,
    "openorca": _build_open_orca,
    "sharegpt-52k": _build_sharegpt_52k,
    "verifiable-corpus": _build_verifiable_corpus,
    "commonsense-qa": _build_commonsense_qa,
    "piqa": _build_piqa,
    "winogrande": _build_winogrande,
    "cosmos-qa": _build_cosmos_qa,
    "boolq": _build_boolq,
    "race": _build_race,
    "social-iqa": _build_social_iqa,
    "qasc": _build_qasc,
    "logiqa": _build_logiqa,
    "aqua-rat-mcqa": _build_aqua_rat_mcqa,
    "gpqa-d": _build_gpqa_d,
    "mmlu-pro": _build_mmlu_pro,
    "aime24": _build_aime24,
    "aime25": _build_aime25,
    "math500": _build_math500,
    "amc23": _build_amc23,
    "mbpp": _build_mbpp,
    "code-bagel": _build_code_bagel,
    "z1-code-reasoning": _build_z1_code_reasoning,
}

DATASET_ALIASES = {
    "openai/gsm8k": "gsm8k",
    "madrylab/gsm8k-platinum": "gsm8k-platinum",
    "567-labs/gsm8k": "gsm8k-567",
    "eleutherai/hendrycks_math": "hendrycks-math",
    "nlile/hendrycks-math-benchmark": "hendrycks-math-benchmark",
    "allenai/math_qa": "math_qa",
    "regisss/math_qa": "math_qa",
    "mu-nlpc/calc-svamp": "calc-svamp",
    "nivektk/math-augmented-dataset": "math-augmented",
    "tatsu-lab/alpaca": "alpaca",
    "aisquared/databricks-dolly-15k": "dolly-15k",
    "databricks/databricks-dolly-15k": "dolly-15k",
    "anthropic/hh-rlhf": "hh-rlhf",
    "lmsys/lmsys-chat-1m": "lmsys-chat-1m",
    "openassistant/oasst1": "oasst1",
    "h2oai/openassistant_oasst1": "oasst1-h2oai",
    "mugivara1/json-mermaid": "json-mermaid",
    "moogin/json-mermaid": "json-mermaid",
    "huggingfaceh4/ultrachat_200k": "ultrachat-200k",
    "open-orca/openorca": "openorca",
    "ryokoai/sharegpt52k": "sharegpt-52k",
    "lasgroup/verifiable-corpus": "verifiable-corpus",
    "commonsense_qa": "commonsense-qa",
    "piqa": "piqa",
    "winogrande": "winogrande",
    "cosmos_qa": "cosmos-qa",
    "boolq": "boolq",
    "race": "race",
    "jet-ai/social_i_qa": "social-iqa",
    "happyeval/social_i_qa-text": "social-iqa",
    "allenai/social_i_qa": "social-iqa",
    "allenai/qasc": "qasc",
    "heka-ai/logiqa": "logiqa",
    "lucasmccabe/logiqa": "logiqa",
    "rikotemaster/aqua-rat-mcqa": "aqua-rat-mcqa",
    "deepmind/aqua_rat": "aqua-rat-mcqa",
    "idavidrein/gpqa-d": "gpqa-d",
    "tiger-lab/mmlu-pro": "mmlu-pro",
    "math-ai/aime24": "aime24",
    "math-ai/aime25": "aime25",
    "math-ai/math500": "math500",
    "math-ai/amc23": "amc23",
    "google-research-datasets/mbpp": "mbpp",
    "replete-ai/code_bagel": "code-bagel",
    "efficientscaling/z1-code-reasoning-107k": "z1-code-reasoning",
}

DEFAULT_TRAIN_DATASETS = [
    "gsm8k",
    "gsm8k-platinum",
    "gsm8k-567",
    "hendrycks-math",
    "hendrycks-math-benchmark",
    "math_qa",
    "calc-svamp",
    "math-augmented",
    "alpaca",
    "dolly-15k",
    "hh-rlhf",
    "lmsys-chat-1m",
    "oasst1",
    "oasst1-h2oai",
    "json-mermaid",
    "ultrachat-200k",
    "openorca",
    "sharegpt-52k",
    "verifiable-corpus",
    "commonsense-qa",
    "piqa",
    "winogrande",
    "cosmos-qa",
    "boolq",
    "race",
    "gpqa-d",
    "mmlu-pro",
    "aime24",
    "aime25",
    "math500",
    "amc23",
    "mbpp",
    "code-bagel",
    "z1-code-reasoning",
]


class GSM8KLikeTask(HFDatasetTask):
    def __init__(self, name, split, subset=None, **kwargs):
        super().__init__([name], split, _gsm8k_like_row, _gsm8k_like_eval, subset=subset, **kwargs)


def resolve_dataset_list(dataset_names):
    if dataset_names is None or dataset_names == "all":
        return list(DEFAULT_TRAIN_DATASETS)
    if isinstance(dataset_names, str):
        dataset_names = [name.strip() for name in dataset_names.split(",") if name.strip()]
    resolved = []
    for name in dataset_names:
        key = DATASET_ALIASES.get(name.lower(), name)
        resolved.append(key)
    unknown = [name for name in resolved if name not in DATASET_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Available: {sorted(DATASET_BUILDERS.keys())}")
    return resolved


def build_sdpo_tasks(dataset_names, split, skip_errors=False):
    resolved = resolve_dataset_list(dataset_names)
    tasks = []
    loaded = []
    errors = []
    for name in resolved:
        builder = DATASET_BUILDERS[name]
        try:
            tasks.extend(builder(split))
            loaded.append(name)
        except Exception as exc:
            if not skip_errors:
                raise
            errors.append((name, exc))
            continue
    if skip_errors and errors:
        error_lines = [f"{name}: {exc}" for name, exc in errors]
        print("[sdpo] Skipped datasets due to load errors:\n" + "\n".join(error_lines))
    return tasks, loaded if skip_errors else resolved


def available_sdpo_datasets():
    return sorted(DATASET_BUILDERS.keys())
