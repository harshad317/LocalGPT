"""
Evaluate nanochat base model on DeepEval benchmarks.

Example:
python -m scripts.deepeval_benchmarks --device cuda --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from nanochat.checkpoint_manager import load_model
try:
    from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
except Exception:  # Back-compat for older nanochat.common
    import torch.distributed as dist

    def print0(s="", **kwargs):
        if int(os.environ.get("RANK", 0)) == 0:
            print(s, **kwargs)

    def autodetect_device_type():
        if torch.cuda.is_available():
            device_type = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
        print0(f"Autodetected device type: {device_type}")
        return device_type

    def compute_init(device_type="cuda"):
        assert device_type in ["cuda", "mps", "cpu"]
        torch.manual_seed(42)
        if device_type == "cuda":
            torch.cuda.manual_seed(42)
        is_ddp = all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        if is_ddp and device_type == "cuda":
            ddp_rank = int(os.environ["RANK"])
            ddp_local_rank = int(os.environ["LOCAL_RANK"])
            ddp_world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(ddp_local_rank)
            device = torch.device("cuda", ddp_local_rank)
            dist.init_process_group(backend="nccl")
            dist.barrier()
            return True, ddp_rank, ddp_local_rank, ddp_world_size, device
        return False, 0, 0, 1, torch.device(device_type)

    def compute_cleanup():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
import nanochat.deepeval_harness as deepeval_harness


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    class_name: str
    display_name: str
    supports_batch: bool = True


BENCHMARK_SPECS = [
    BenchmarkSpec("mmlu", "MMLU", "MMLU"),
    BenchmarkSpec("hellaswag", "HellaSwag", "HellaSwag"),
    BenchmarkSpec("big-bench-hard", "BigBenchHard", "Big-Bench Hard"),
    BenchmarkSpec("drop", "DROP", "DROP"),
    BenchmarkSpec("truthfulqa", "TruthfulQA", "TruthfulQA"),
    BenchmarkSpec("humaneval", "HumanEval", "HumanEval", supports_batch=False),
    BenchmarkSpec("ifeval", "IFEval", "IFEval"),
    BenchmarkSpec("squad", "SQuAD", "SQuAD"),
    BenchmarkSpec("gsm8k", "GSM8K", "GSM8K", supports_batch=False),
    BenchmarkSpec("mathqa", "MathQA", "MathQA"),
    BenchmarkSpec("logiqa", "LogiQA", "LogiQA"),
    BenchmarkSpec("boolq", "BoolQ", "BoolQ"),
    BenchmarkSpec("arc", "ARC", "ARC"),
    BenchmarkSpec("bbq", "BBQ", "BBQ"),
    BenchmarkSpec("lambada", "LAMBADA", "LAMBADA"),
    BenchmarkSpec("winogrande", "Winogrande", "Winogrande"),
]

_SPEC_BY_KEY = {spec.key: spec for spec in BENCHMARK_SPECS}

_ALIASES = {
    "bbh": "big-bench-hard",
    "big_bench_hard": "big-bench-hard",
    "bigbenchhard": "big-bench-hard",
    "big-benchhard": "big-bench-hard",
    "human-eval": "humaneval",
    "human_eval": "humaneval",
    "human eval": "humaneval",
    "gsm-8k": "gsm8k",
    "gsm_8k": "gsm8k",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _selected_specs(selection: Optional[str]) -> list[BenchmarkSpec]:
    if not selection or selection.strip().lower() == "all":
        return BENCHMARK_SPECS
    raw = [_normalize_name(item) for item in selection.split(",") if item.strip()]
    selected = []
    unknown = []
    for name in raw:
        name = _ALIASES.get(name, name)
        spec = _SPEC_BY_KEY.get(name)
        if spec is None:
            unknown.append(name)
            continue
        selected.append(spec)
    if unknown:
        known = ", ".join(sorted(_SPEC_BY_KEY.keys()))
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown)}. Known: {known}")
    return selected


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: autodetect)")
    parser.add_argument("--model-tag", type=str, default=None, help="Checkpoint tag (e.g., d20)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (e.g., 1000)")
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated list or 'all'")
    parser.add_argument("--list-benchmarks", action="store_true", help="List available benchmarks and exit")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for supported benchmarks")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (default: disabled)")
    parser.add_argument("--stop", type=str, default=None, help="Comma-separated stop strings")
    parser.add_argument("--human-eval-n", type=int, default=None, help="HumanEval number of samples (default: library)")
    parser.add_argument("--human-eval-k", type=int, default=1, help="HumanEval pass@k value")
    parser.add_argument(
        "--squad-eval-model",
        type=str,
        default="gpt-4.1",
        help="SQuAD evaluation model (string or 'local')",
    )
    parser.add_argument("--out", type=str, default=None, help="Write JSON results to this path")
    return parser.parse_args()


def _format_model_name(model_tag: Optional[str], step: Optional[int]) -> str:
    parts = ["nanochat-base"]
    if model_tag:
        parts.append(model_tag)
    if step is not None:
        parts.append(f"step{step}")
    return "-".join(parts)


def _build_benchmark(spec: BenchmarkSpec, cls, args, model):
    if spec.key == "humaneval":
        kwargs = {}
        if args.human_eval_n is not None:
            kwargs["n"] = args.human_eval_n
        return cls(**kwargs)
    if spec.key == "squad":
        if args.squad_eval_model.lower() in {"local", "self"}:
            return cls(evaluation_model=model)
        return cls(evaluation_model=args.squad_eval_model)
    return cls()


def main() -> int:
    args = _parse_args()

    try:
        deepeval_harness.ensure_deepeval()
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        from deepeval import benchmarks as de_benchmarks
    except Exception as exc:
        print(f"Failed to import deepeval benchmarks: {exc}", file=sys.stderr)
        return 1

    if args.list_benchmarks:
        for spec in BENCHMARK_SPECS:
            print(f"{spec.key}\t{spec.display_name}")
        return 0

    if args.human_eval_n is not None and args.human_eval_k > args.human_eval_n:
        print("--human-eval-k must be <= --human-eval-n", file=sys.stderr)
        return 1

    try:
        selected = _selected_specs(args.benchmarks)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    device_type = args.device or autodetect_device_type()
    is_ddp, ddp_rank, _, _, device = compute_init(device_type=device_type)

    if is_ddp and ddp_rank != 0:
        compute_cleanup()
        return 0

    model, tokenizer, _ = load_model(
        "base",
        device=device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )

    stop = [s for s in (args.stop.split(",") if args.stop else []) if s]
    llm = deepeval_harness.NanochatDeepEvalLLM(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_name=_format_model_name(args.model_tag, args.step),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stop=stop,
    )

    if args.squad_eval_model == "gpt-4.1" and not os.environ.get("OPENAI_API_KEY"):
        print0("Warning: SQuAD default evaluation model is gpt-4.1 but OPENAI_API_KEY is not set.")

    results = {}
    for spec in selected:
        cls = getattr(de_benchmarks, spec.class_name, None)
        if cls is None:
            print0(f"Skipping {spec.display_name}; {spec.class_name} not found in deepeval.benchmarks.")
            results[spec.key] = {"error": "benchmark class not found"}
            continue

        print0(f"Evaluating {spec.display_name}...")
        benchmark = _build_benchmark(spec, cls, args, llm)
        eval_kwargs = {}
        if spec.supports_batch and args.batch_size:
            eval_kwargs["batch_size"] = args.batch_size
        if spec.key == "humaneval":
            eval_kwargs["k"] = args.human_eval_k

        start = time.time()
        try:
            benchmark.evaluate(llm, **eval_kwargs)
            overall_score = _to_jsonable(getattr(benchmark, "overall_score", None))
            results[spec.key] = {
                "overall_score": overall_score,
                "seconds": round(time.time() - start, 2),
            }
            print0(f"Done {spec.display_name}: {overall_score}")
        except Exception as exc:
            results[spec.key] = {
                "error": str(exc),
                "seconds": round(time.time() - start, 2),
            }
            print0(f"Failed {spec.display_name}: {exc}")

    if ddp_rank == 0:
        print(json.dumps(results, indent=2, sort_keys=True))
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Wrote results to {out_path}")

    compute_cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
