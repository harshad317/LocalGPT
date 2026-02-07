"""
Helpers to run EleutherAI's lm-evaluation-harness on nanochat models.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

_IMPORT_ERROR = None
try:
    from lm_eval.api.model import LM
    from lm_eval import utils as lm_utils
    from lm_eval.tasks import TaskManager
except Exception as exc:  # pragma: no cover - guarded for optional dependency
    LM = None  # type: ignore[assignment]
    TaskManager = None  # type: ignore[assignment]
    lm_utils = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def ensure_lm_eval() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "lm-evaluation-harness is required. Install it with "
            "`pip install lm-eval` or `pip install -e /path/to/lm-evaluation-harness`."
        ) from _IMPORT_ERROR


_MATH_DESC_KEYWORDS = (
    "math",
    "arithmetic",
    "algebra",
    "geometry",
    "calculus",
    "number theory",
    "quantitative",
    "numerical",
    "numeric",
)
_MATH_TAG_KEYWORDS = ("math", "arithmetic", "gsm")


@dataclass(frozen=True)
class TaskFamilyMeta:
    description: str
    languages: str


def _extract_family(cell: str) -> str:
    match = re.search(r"\[(.*?)\]", cell)
    return match.group(1) if match else cell.strip()


def parse_task_families(tasks_root: Path) -> dict[str, TaskFamilyMeta]:
    readme_path = tasks_root / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"lm_eval tasks README not found at {readme_path}")
    families: dict[str, TaskFamilyMeta] = {}
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().split("|")[1:-1]]
        if len(parts) < 3 or parts[0] == "Task Family":
            continue
        family = _extract_family(parts[0])
        families[family] = TaskFamilyMeta(description=parts[1], languages=parts[2])
    return families


def _is_math_task(family: str, meta: TaskFamilyMeta, tags: Iterable[str]) -> bool:
    text = f"{family} {meta.description}".lower()
    if any(keyword in text for keyword in _MATH_DESC_KEYWORDS):
        return True
    tag_text = " ".join(tags).lower()
    if any(keyword in tag_text for keyword in _MATH_TAG_KEYWORDS):
        return True
    return False


def select_english_or_math_tasks(task_manager: "TaskManager", tasks_root: Path) -> list[str]:
    ensure_lm_eval()
    families = parse_task_families(tasks_root)
    selected: list[str] = []
    for task in task_manager.all_subtasks:
        info = task_manager.task_index.get(task, {})
        yaml_path = info.get("yaml_path")
        if not yaml_path or yaml_path == -1:
            continue
        yaml_path = Path(yaml_path)
        try:
            family = yaml_path.resolve().relative_to(tasks_root.resolve()).parts[0]
        except Exception:
            continue
        meta = families.get(family)
        if meta is None:
            continue
        config = lm_utils.load_yaml_config(yaml_path=str(yaml_path), mode="simple")
        tags = config.get("tag") or []
        if isinstance(tags, str):
            tags = [tags]
        is_english = "English" in meta.languages
        is_math = _is_math_task(family, meta, tags)
        if is_english or is_math:
            selected.append(task)
    return sorted(set(selected))


if LM is not None:

    class NanochatLM(LM):
        tokenizer_name = "nanochat"

        def __init__(
            self,
            model,
            tokenizer,
            device: torch.device,
            max_length: Optional[int] = None,
            default_max_gen_toks: int = 256,
            rank: int = 0,
            world_size: int = 1,
        ) -> None:
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self._rank = rank
            self._world_size = world_size
            self.bos_token_id = tokenizer.get_bos_token_id()
            self.max_length = max_length or getattr(model.config, "sequence_len", 1024)
            self.default_max_gen_toks = default_max_gen_toks

        def _truncate_to_maxlen(self, context_ids: list[int], continuation_ids: list[int]):
            total_len = len(context_ids) + len(continuation_ids)
            if total_len <= self.max_length:
                return context_ids, continuation_ids
            overflow = total_len - self.max_length
            # Keep at least one context token.
            if len(context_ids) > 1:
                trim_ctx = min(overflow, len(context_ids) - 1)
                context_ids = context_ids[trim_ctx:]
                overflow -= trim_ctx
            if overflow > 0:
                continuation_ids = continuation_ids[overflow:]
            return context_ids, continuation_ids

        def _loglikelihood_tokens(
            self, context_ids: list[int], continuation_ids: list[int]
        ) -> tuple[float, bool]:
            context_ids, continuation_ids = self._truncate_to_maxlen(
                context_ids, continuation_ids
            )
            if not continuation_ids:
                return 0.0, True
            tokens = context_ids + continuation_ids
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                logits = self.model.forward(input_ids)  # (1, T, V)
            start = len(context_ids)
            logits = logits[0, start - 1 : start - 1 + len(continuation_ids)]
            log_probs = F.log_softmax(logits, dim=-1)
            cont = torch.tensor(continuation_ids, dtype=torch.long, device=self.device)
            selected = log_probs.gather(-1, cont.unsqueeze(-1)).squeeze(-1)
            total_logprob = selected.sum().item()
            greedy = torch.argmax(logits, dim=-1)
            is_greedy = bool(torch.equal(greedy, cont))
            return total_logprob, is_greedy

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            from tqdm import tqdm

            res = []
            for req in tqdm(requests, disable=disable_tqdm or (self.rank != 0)):
                context, continuation = req.arguments
                context_ids = self.tokenizer.encode(
                    context, prepend=self.bos_token_id
                )
                continuation_ids = self.tokenizer.encode(continuation)
                logprob, is_greedy = self._loglikelihood_tokens(
                    context_ids, continuation_ids
                )
                res.append((logprob, is_greedy))
                self.cache_hook.add_partial(
                    "loglikelihood", (context, continuation), (logprob, is_greedy)
                )
            return res

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            from tqdm import tqdm

            res = []
            for req in tqdm(requests, disable=disable_tqdm or (self.rank != 0)):
                (string,) = req.args
                tokens = self.tokenizer.encode(string)
                total = 0.0
                windows = map(
                    lm_utils.make_disjoint_window,
                    lm_utils.get_rolling_token_windows(
                        token_list=tokens,
                        prefix_token=self.bos_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
                for context_ids, continuation_ids in windows:
                    logprob, _ = self._loglikelihood_tokens(
                        context_ids, continuation_ids
                    )
                    total += logprob
                res.append(total)
                self.cache_hook.add_partial(
                    "loglikelihood_rolling", (string,), total
                )
            return res

        def _sample_next_token(
            self, logits: torch.Tensor, temperature: float, top_k: Optional[int]
        ) -> int:
            if temperature <= 0:
                return int(torch.argmax(logits, dim=-1).item())
            logits = logits / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                vals, idx = torch.topk(logits, k)
                probs = F.softmax(vals, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                return int(idx.gather(-1, choice).item())
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())

        def _generate_one(self, context: str, gen_kwargs: dict) -> str:
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks")
            if max_gen_toks is None and "max_length" in gen_kwargs:
                max_gen_toks = max(
                    0, int(gen_kwargs["max_length"]) - len(self.tokenizer.encode(context))
                )
            if max_gen_toks is None:
                max_gen_toks = self.default_max_gen_toks

            do_sample = gen_kwargs.get("do_sample")
            temperature = float(gen_kwargs.get("temperature", 0.0))
            if do_sample is False:
                temperature = 0.0
            top_k = gen_kwargs.get("top_k")

            context_ids = self.tokenizer.encode(
                context, prepend=self.bos_token_id
            )
            tokens = list(context_ids)
            generated_ids: list[int] = []
            generated_text = ""

            for _ in range(int(max_gen_toks)):
                window = tokens[-self.max_length :]
                input_ids = torch.tensor([window], dtype=torch.long, device=self.device)
                with torch.inference_mode():
                    logits = self.model.forward(
                        input_ids, logits_positions=input_ids.size(1) - 1
                    )[0]
                next_id = self._sample_next_token(logits, temperature, top_k)
                tokens.append(next_id)
                generated_ids.append(next_id)

                if until:
                    generated_text = self.tokenizer.decode(generated_ids)
                    stop_positions = [
                        generated_text.find(s) for s in until if s and s in generated_text
                    ]
                    if stop_positions:
                        stop_idx = min(stop_positions)
                        return generated_text[:stop_idx]

            return self.tokenizer.decode(generated_ids)

        def generate_until(self, requests, disable_tqdm: bool = False):
            from tqdm import tqdm

            res = []
            for req in tqdm(requests, disable=disable_tqdm or (self.rank != 0)):
                context, gen_kwargs = req.arguments
                gen_kwargs = gen_kwargs or {}
                out = self._generate_one(context, gen_kwargs)
                res.append(out)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), out
                )
            return res

else:
    NanochatLM = None  # type: ignore[assignment]
