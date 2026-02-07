"""
Evaluate nanochat base model on English-language tasks plus math tasks via lm-evaluation-harness.

Example:
python -m scripts.lm_eval_math_en --device cuda --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from nanochat.checkpoint_manager import load_model
try:
    from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
except Exception:  # Back-compat for older nanochat.common
    import os
    import torch.distributed as dist

    def autodetect_device_type():
        if torch.cuda.is_available():
            device_type = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
        print(f"Autodetected device type: {device_type}")
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
from nanochat.lm_eval_harness import (
    NanochatLM,
    ensure_lm_eval,
    select_english_or_math_tasks,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | mps (default: autodetect)")
    parser.add_argument("--model-tag", type=str, default=None, help="Checkpoint tag (e.g., d20)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (e.g., 1000)")
    parser.add_argument("--limit", type=float, default=None, help="Limit examples per task (int or fraction)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size hint for lm-eval (default: 1)")
    parser.add_argument("--out", type=str, default=None, help="Write results JSON to this path")
    parser.add_argument("--list-tasks", action="store_true", help="Print selected tasks and exit")
    parser.add_argument("--confirm-unsafe-code", action="store_true", help="Allow tasks marked unsafe")
    parser.add_argument(
        "--include-unitxt",
        action="store_true",
        help="Include unitxt tasks (requires `pip install unitxt`).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task list. If set, overrides English math selection.",
    )
    return parser.parse_args()


def _unitxt_available() -> bool:
    try:
        import unitxt  # noqa: F401

        return True
    except Exception:
        return False


def _eval_metrics_available() -> bool:
    try:
        import evaluate  # noqa: F401
        import bert_score  # noqa: F401
        import bleurt  # noqa: F401

        return True
    except Exception:
        return False


def _filter_dataset_script_tasks(task_manager, task_list: list[str]):
    try:
        import datasets
        from lm_eval import utils as lm_utils
    except Exception:
        return task_list, []

    filtered: list[str] = []
    removed: list[str] = []

    for task in task_list:
        info = task_manager.task_index.get(task, {})
        yaml_path = info.get("yaml_path")
        if not yaml_path or yaml_path == -1:
            filtered.append(task)
            continue
        try:
            config = lm_utils.load_yaml_config(yaml_path=str(yaml_path), mode="simple")
        except Exception:
            filtered.append(task)
            continue
        if not isinstance(config, dict):
            filtered.append(task)
            continue
        dataset_path = config.get("dataset_path")
        if not dataset_path:
            filtered.append(task)
            continue
        if isinstance(dataset_path, Path):
            dataset_path = str(dataset_path)
        if not isinstance(dataset_path, str):
            filtered.append(task)
            continue

        dataset_kwargs = {}
        raw_kwargs = config.get("dataset_kwargs")
        if isinstance(raw_kwargs, dict):
            if raw_kwargs.get("name"):
                dataset_kwargs["name"] = raw_kwargs["name"]
            elif raw_kwargs.get("config_name"):
                dataset_kwargs["name"] = raw_kwargs["config_name"]
        for key in ("dataset_name", "dataset_config", "dataset_config_name"):
            if key in config and config[key] and "name" not in dataset_kwargs:
                dataset_kwargs["name"] = config[key]

        try:
            datasets.load_dataset_builder(dataset_path, **dataset_kwargs)
        except Exception as exc:
            if "Dataset scripts are no longer supported" in str(exc):
                removed.append(task)
                continue
        filtered.append(task)

    return filtered, removed


def _filter_family(task_manager, tasks_root: Path, task_list: list[str], family: str):
    filtered = []
    removed = []
    for task in task_list:
        info = task_manager.task_index.get(task, {})
        yaml_path = info.get("yaml_path")
        if not yaml_path or yaml_path == -1:
            filtered.append(task)
            continue
        try:
            rel = Path(yaml_path).resolve().relative_to(tasks_root.resolve())
        except Exception:
            filtered.append(task)
            continue
        if rel.parts and rel.parts[0] == family:
            removed.append(task)
            continue
        filtered.append(task)
    return filtered, removed


def main() -> int:
    args = _parse_args()

    try:
        ensure_lm_eval()
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    import lm_eval
    import lm_eval.tasks
    from lm_eval.tasks import TaskManager
    from lm_eval.evaluator import simple_evaluate

    task_manager = TaskManager()
    tasks_root = Path(lm_eval.tasks.__file__).resolve().parent

    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_list = select_english_or_math_tasks(task_manager, tasks_root)

    if not args.include_unitxt:
        task_list, removed = _filter_family(task_manager, tasks_root, task_list, "unitxt")
        if removed:
            if _unitxt_available():
                msg = (
                    f"Skipping {len(removed)} unitxt task(s); pass --include-unitxt "
                    "to include them."
                )
            else:
                msg = (
                    f"Skipping {len(removed)} unitxt task(s); install unitxt to include them."
                )
            print(msg, file=sys.stderr)

    if not _eval_metrics_available():
        task_list, removed = _filter_family(task_manager, tasks_root, task_list, "careqa")
        if removed:
            print(
                f"Skipping {len(removed)} careqa task(s); install evaluate, bert-score, "
                "and bleurt to include them.",
                file=sys.stderr,
            )

    task_list, removed = _filter_dataset_script_tasks(task_manager, task_list)
    if removed:
        print(
            f"Skipping {len(removed)} task(s) that require dataset scripts; "
            "pin `datasets<3` or update tasks to include them.",
            file=sys.stderr,
        )

    if not task_list:
        print("No tasks matched the English or math filter. Check lm-eval version.", file=sys.stderr)
        return 1

    if args.list_tasks:
        print("\n".join(task_list))
        return 0

    device_type = args.device or autodetect_device_type()
    is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type=device_type)

    model, tokenizer, _ = load_model(
        "base",
        device=device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )

    lm = NanochatLM(
        model,
        tokenizer,
        device=device,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )

    try:
        results = simple_evaluate(
            model=lm,
            tasks=task_list,
            limit=args.limit,
            batch_size=args.batch_size,
            task_manager=task_manager,
            confirm_run_unsafe_code=args.confirm_unsafe_code,
        )

        if results is None:
            return 0

        if ddp_rank == 0:
            print(json.dumps(results["results"], indent=2, sort_keys=True))

            if args.out:
                out_path = Path(args.out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(f"Wrote results to {out_path}")
        return 0
    finally:
        compute_cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
