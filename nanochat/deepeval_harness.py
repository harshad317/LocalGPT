"""
Helpers to run DeepEval benchmarks on nanochat models.
"""

from __future__ import annotations

from typing import Optional

from contextlib import nullcontext

import torch

_IMPORT_ERROR = None
try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except Exception as exc:  # pragma: no cover - optional dependency
    DeepEvalBaseLLM = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def ensure_deepeval() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "deepeval is required. Install it with `pip install deepeval`."
        ) from _IMPORT_ERROR


def _truncate_on_stop(text: str, stop: Optional[list[str]]) -> str:
    if not stop:
        return text
    earliest = None
    for token in stop:
        if not token:
            continue
        idx = text.find(token)
        if idx == -1:
            continue
        if earliest is None or idx < earliest:
            earliest = idx
    return text if earliest is None else text[:earliest]


if DeepEvalBaseLLM is not None:

    class NanochatDeepEvalLLM(DeepEvalBaseLLM):
        def __init__(
            self,
            model,
            tokenizer,
            device: torch.device,
            model_name: str,
            max_new_tokens: int = 256,
            temperature: float = 0.0,
            top_k: Optional[int] = None,
            seed: int = 42,
            stop: Optional[list[str]] = None,
        ) -> None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.model_name = model_name
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.top_k = top_k
            self.seed = seed
            self.stop = stop or []
            self.bos_token_id = tokenizer.get_bos_token_id()
            self.max_length = getattr(model.config, "sequence_len", 1024)
            self.model.eval()

        def get_model_name(self):
            return self.model_name

        def load_model(self):
            return self.model

        def _generate_text(
            self,
            prompt: str,
            *,
            temperature: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            top_k: Optional[int] = None,
            stop: Optional[list[str]] = None,
            seed: Optional[int] = None,
        ) -> str:
            prompt_ids = self.tokenizer.encode(prompt, prepend=self.bos_token_id)
            max_new_tokens = max_new_tokens or self.max_new_tokens
            temperature = self.temperature if temperature is None else temperature
            top_k = self.top_k if top_k is None else top_k
            stop = self.stop if stop is None else stop

            remaining = self.max_length - len(prompt_ids)
            if remaining <= 0:
                return ""
            max_new_tokens = min(max_new_tokens, remaining)

            generated: list[int] = []
            autocast_ctx = (
                torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                for token in self.model.generate(
                    prompt_ids,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=self.seed if seed is None else seed,
                ):
                    generated.append(token)
                    if stop:
                        text = self.tokenizer.decode(generated)
                        if any(s and s in text for s in stop):
                            break

            text = self.tokenizer.decode(generated)
            return _truncate_on_stop(text, stop)

        def generate(self, prompt: str) -> str:
            return self._generate_text(prompt)

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def batch_generate(self, prompts: list[str]) -> list[str]:
            return [self.generate(prompt) for prompt in prompts]

        def generate_samples(self, prompt: str, n: int, temperature: float):
            return [
                self._generate_text(
                    prompt, temperature=temperature, seed=self.seed + i
                )
                for i in range(n)
            ]
