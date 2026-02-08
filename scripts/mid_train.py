"""
Midtrain the model. Same as pretraining but simpler.
Run as:

python -m scripts.mid_train

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device-batch-size=16
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import random
import time
from contextlib import nullcontext

import wandb
import torch

try:
    from nanochat.common import (
        compute_init,
        compute_cleanup,
        print0,
        DummyWandb,
        get_base_dir,
        autodetect_device_type,
    )
except ImportError:
    # Back-compat: older `nanochat.common` may not define compute_init yet.
    import torch.distributed as dist
    import nanochat.common as _common

    def _maybe_attr(name):
        return getattr(_common, name, None)

    print0 = _maybe_attr("print0")
    if print0 is None:
        def print0(s="", **kwargs):
            ddp_rank = int(os.environ.get("RANK", 0))
            if ddp_rank == 0:
                print(s, **kwargs)

    DummyWandb = _maybe_attr("DummyWandb")
    if DummyWandb is None:
        class DummyWandb:
            def log(self, *args, **kwargs):
                pass
            def finish(self):
                pass

    get_base_dir = _maybe_attr("get_base_dir")
    if get_base_dir is None:
        def get_base_dir():
            if os.environ.get("NANOCHAT_BASE_DIR"):
                nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
            else:
                nanochat_dir = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
            os.makedirs(nanochat_dir, exist_ok=True)
            return nanochat_dir

    autodetect_device_type = _maybe_attr("autodetect_device_type")
    if autodetect_device_type is None:
        def autodetect_device_type():
            if torch.cuda.is_available():
                device_type = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device_type = "mps"
            else:
                device_type = "cpu"
            print0(f"Autodetected device type: {device_type}")
            return device_type

    compute_init = _maybe_attr("compute_init")
    if compute_init is None:
        def compute_init(device_type="cuda"):
            assert device_type in ["cuda", "mps", "cpu"], "Invalid device type"
            if device_type == "cuda":
                assert torch.cuda.is_available(), (
                    "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
                )
            if device_type == "mps":
                assert getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(), (
                    "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
                )

            torch.manual_seed(42)
            if device_type == "cuda":
                torch.cuda.manual_seed(42)
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass

            is_ddp_requested = all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
            ddp_rank = int(os.environ.get("RANK", 0))
            ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
            ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))

            if is_ddp_requested and device_type == "cuda":
                torch.cuda.set_device(ddp_local_rank)
                device = torch.device("cuda", ddp_local_rank)
                dist.init_process_group(backend="nccl")
                dist.barrier()
            else:
                device = torch.device(device_type)

            return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

    compute_cleanup = _maybe_attr("compute_cleanup")
    if compute_cleanup is None:
        def compute_cleanup():
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
from nanochat.early_stopping import EarlyStopping
from nanochat.engine import Engine
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
import torch.distributed as dist

# Back-compat for older `tasks.common` APIs.
# Some forks/checkouts may not export Task helpers needed by task modules.
def _ensure_tasks_common_api():
    try:
        import random
        import tasks.common as tc
    except Exception:
        return

    if not hasattr(tc, "Task"):
        if hasattr(tc, "BaseTask"):
            tc.Task = tc.BaseTask
        elif hasattr(tc, "TaskBase"):
            tc.Task = tc.TaskBase
        else:
            class Task:
                def __init__(self, start=0, stop=None, step=1):
                    assert start >= 0
                    assert stop is None or stop >= start
                    assert step >= 1
                    self.start = start
                    self.stop = stop
                    self.step = step

                @property
                def eval_type(self):
                    raise NotImplementedError

                def num_examples(self):
                    raise NotImplementedError

                def get_example(self, index):
                    raise NotImplementedError

                def __len__(self):
                    start = self.start
                    stop = self.num_examples() if self.stop is None else self.stop
                    step = self.step
                    span = stop - start
                    return (span + step - 1) // step

                def __getitem__(self, index: int):
                    physical_index = self.start + index * self.step
                    return self.get_example(physical_index)

                def evaluate(self, problem, completion):
                    raise NotImplementedError

            tc.Task = Task

    if not hasattr(tc, "TaskMixture"):
        class TaskMixture(tc.Task):
            def __init__(self, tasks, **kwargs):
                super().__init__(**kwargs)
                self.tasks = tasks
                self.lengths = [len(task) for task in self.tasks]
                self.num_conversations = sum(self.lengths)
                self.index_map = []
                for task_idx, task_length in enumerate(self.lengths):
                    for local_idx in range(task_length):
                        self.index_map.append((task_idx, local_idx))
                rng = random.Random(42)
                rng.shuffle(self.index_map)

            @property
            def eval_type(self):
                eval_types = {t.eval_type for t in self.tasks}
                if len(eval_types) != 1:
                    raise ValueError(f"TaskMixture contains mixed eval types: {sorted(eval_types)}")
                return next(iter(eval_types))

            def num_examples(self):
                return self.num_conversations

            def get_example(self, index):
                task_idx, local_idx = self.index_map[index]
                return self.tasks[task_idx][local_idx]

        tc.TaskMixture = TaskMixture

    if not hasattr(tc, "TaskSequence"):
        class TaskSequence(tc.Task):
            def __init__(self, tasks, **kwargs):
                super().__init__(**kwargs)
                self.tasks = tasks
                self.lengths = [len(task) for task in self.tasks]
                self.num_conversations = sum(self.lengths)

            @property
            def eval_type(self):
                eval_types = {t.eval_type for t in self.tasks}
                if len(eval_types) != 1:
                    raise ValueError(f"TaskSequence contains mixed eval types: {sorted(eval_types)}")
                return next(iter(eval_types))

            def num_examples(self):
                return self.num_conversations

            def get_example(self, index):
                for task_idx, task_length in enumerate(self.lengths):
                    if index < task_length:
                        return self.tasks[task_idx][index]
                    index -= task_length
                raise IndexError(index)

        tc.TaskSequence = TaskSequence

    if not hasattr(tc, "render_mc"):
        def render_mc(question, letters, choices):
            query = f"Multiple Choice question: {question}\n"
            query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
            query += "\nRespond only with the letter of the correct answer."
            return query

        tc.render_mc = render_mc


_ensure_tasks_common_api()
from tasks.common import Task, TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU, MMLUSubjects, MMLU_SUBJECT_GROUPS
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.hellaswag import HellaSwag
from tasks.gpqa import GPQA
from tasks.hendrycks_math import HendrycksMath
from tasks.aime import AIME2024, AIME2025
from tasks.alpaca import Alpaca
from tasks.xlam_function_calling import XLAMFunctionCalling
from tasks.triviaqa import TriviaQA
from tasks.science_qa import SciQ, OpenBookQA
from tasks.spellingbee import SimpleSpelling, SpellingBee
from tasks.open_thoughts import OpenThoughts2
from tasks.skunkworks_reasoning import SkunkworksReasoning
from tasks.open_math_instruct import OpenMathInstruct2
from tasks.hermes_function_calling import HermesFunctionCalling
from tasks.mmlu_pro import MMLUPro
from tasks.hle import HLE
from tasks.humaneval import HumanEval
from tasks.magpie_reasoning import MagpieReasoning
from tasks.natural_reasoning import NaturalReasoning
from tasks.numina_math_qwq import NuminaMathQwQ
from tasks.function_calling_sharegpt import FunctionCallingShareGPT
from tasks.ifeval import IFEval
from tasks.bfcl_v3 import build_bfcl_v3_benchmark
from tasks.sdpo_datasets import build_sdpo_tasks

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Midtrain the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = use --num-epochs)")
parser.add_argument("--num-epochs", type=int, default=15, help="number of epochs to run when --num-iterations < 0")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="initial LR as fraction of base LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=1000, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument(
    "--eval-every-epoch",
    action="store_true",
    help="evaluate val bpb at the end of each epoch (plus an initial eval at step 0)",
)
parser.add_argument("--eval-profile", type=str, default="default", choices=["default", "benchmarks", "both"], help="validation dataset profile")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
# Benchmark evaluation
parser.add_argument("--bench-eval", action="store_true", help="run benchmark task evals at each eval step")
parser.add_argument("--bench-eval-max-problems", type=int, default=None, help="max problems per benchmark (None = all)")
parser.add_argument("--bench-eval-batch-size", type=int, default=8, help="batch size for categorical benchmark evals")
parser.add_argument("--bench-eval-num-samples", type=int, default=1, help="num samples per benchmark problem")
parser.add_argument("--bench-eval-max-new-tokens", type=int, default=512, help="max tokens to generate per benchmark problem")
parser.add_argument("--bench-eval-temperature", type=float, default=0.0, help="temperature for benchmark generation")
parser.add_argument("--bench-eval-top-k", type=int, default=50, help="top-k sampling for benchmark generation")
# Early stopping (based on val bpb)
parser.add_argument("--early-stop-patience", type=int, default=2, help="early stop after N non-improving evals (0 disables)")
parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="min improvement in bpb to reset patience")
parser.add_argument(
    "--early-stop-metric",
    type=str,
    default="val_bpb",
    choices=["val_bpb", "bench"],
    help="metric to monitor for early stopping: val_bpb (lower is better) or bench (higher is better)",
)
# Data
parser.add_argument(
    "--dataset-profile",
    type=str,
    default="default",
    choices=["default", "benchmarks", "both"],
    help="dataset mixture profile: default (original), benchmarks (adds more reasoning/math/tool-style data), or both",
)
parser.add_argument("--upweight-mmlu-pro", type=int, default=2, help="repeat factor for MMLU-Pro in benchmarks profile")
parser.add_argument("--upweight-gsm8k", type=int, default=2, help="repeat factor for GSM8K in benchmarks profile")
parser.add_argument("--upweight-hendrycks", type=int, default=2, help="repeat factor for HendrycksMath subjects in benchmarks profile")
parser.add_argument("--upweight-mmlu", type=int, default=1, help="repeat factor for MMLU in benchmarks profile")
parser.add_argument("--upweight-mmlu-physics", type=int, default=1, help="repeat factor for MMLU physics subjects")
parser.add_argument("--upweight-mmlu-biology", type=int, default=1, help="repeat factor for MMLU biology subjects")
parser.add_argument("--upweight-mmlu-engineering", type=int, default=1, help="repeat factor for MMLU engineering subjects")
parser.add_argument("--upweight-mmlu-cs", type=int, default=1, help="repeat factor for MMLU computer science subjects")
parser.add_argument("--upweight-mmlu-it", type=int, default=1, help="repeat factor for MMLU IT/security subjects")
parser.add_argument(
    "--mmlu-subject-upweight-split",
    type=str,
    default="auxiliary_train",
    choices=["auxiliary_train", "dev", "validation", "test"],
    help="split to source MMLU subject upweighting from (auxiliary_train avoids eval leakage; dev/validation/test include eval data)",
)
parser.add_argument("--upweight-arc", type=int, default=1, help="repeat factor for ARC (both Easy/Challenge) in benchmarks profile")
parser.add_argument("--upweight-hellaswag", type=int, default=1, help="repeat factor for HellaSwag in benchmarks profile")
parser.add_argument("--upweight-triviaqa", type=int, default=1, help="repeat factor for TriviaQA in benchmarks profile")
parser.add_argument("--upweight-gpqa", type=int, default=3, help="repeat factor for GPQA in benchmarks profile (default preserves prior 3x)")
parser.add_argument("--phase2-start-epoch", type=int, default=-1, help="enable two-phase schedule at this epoch (1-based). <=0 disables")
parser.add_argument("--phase2-upweight-multiplier", type=int, default=1, help="multiplier for upweighted tasks in phase 2 (>=1)")
# Base-data replay (anti-forgetting)
parser.add_argument("--base-replay-mix", type=float, default=0.0, help="fraction of training micro-batches sampled from base pretraining data (0 disables)")
parser.add_argument("--base-replay-datasets", type=str, default="", help="comma-separated base dataset names (defaults to NANOCHAT_BASE_DATASETS / defaults)")
parser.add_argument("--base-replay-buffer-size", type=int, default=1000, help="buffer size for base replay best-fit packing")
parser.add_argument("--base-replay-tokenizer-batch-size", type=int, default=128, help="tokenizer batch size for base replay loader")
parser.add_argument(
    "--extra-jsonl",
    action="append",
    default=[],
    help="additional JSONL conversation files (repeatable). Each line must be a JSON array of {role,content} messages.",
)
# Output
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--dry-run", action="store_true", help="log to wandb but skip checkpoints/report")
args = parser.parse_args()

if args.eval_every_epoch and args.eval_every > 0:
    print0("Eval every epoch enabled; disabling step-based eval.")
    args.eval_every = -1
for name in (
    "upweight_mmlu_pro",
    "upweight_gsm8k",
    "upweight_hendrycks",
    "upweight_mmlu",
    "upweight_mmlu_physics",
    "upweight_mmlu_biology",
    "upweight_mmlu_engineering",
    "upweight_mmlu_cs",
    "upweight_mmlu_it",
    "upweight_arc",
    "upweight_hellaswag",
    "upweight_triviaqa",
    "upweight_gpqa",
    "phase2_upweight_multiplier",
):
    if getattr(args, name) < 1:
        raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
if not (0.0 <= args.base_replay_mix <= 1.0):
    raise ValueError("--base-replay-mix must be between 0.0 and 1.0")
if args.base_replay_buffer_size < 1:
    raise ValueError("--base-replay-buffer-size must be >= 1")
if args.base_replay_tokenizer_batch_size < 1:
    raise ValueError("--base-replay-tokenizer-batch-size must be >= 1")
if args.phase2_start_epoch > 0 and args.phase2_start_epoch > args.num_epochs:
    print0(
        f"Warning: --phase2-start-epoch ({args.phase2_start_epoch}) exceeds --num-epochs "
        f"({args.num_epochs}); disabling phase2 schedule."
    )
    args.phase2_start_epoch = -1

user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
if args.num_epochs < 1:
    raise ValueError("--num-epochs must be >= 1")
if args.num_iterations > 0 and args.num_epochs != 1:
    print0("Warning: --num-iterations is set, so --num-epochs will be ignored.")
if args.early_stop_patience > 0 and (args.eval_every < 0 and not args.eval_every_epoch):
    print0("Warning: early stopping is enabled but eval is disabled; no early stop will trigger.")
if args.early_stop_metric == "bench" and not args.bench_eval:
    print0("Warning: early stopping on benchmarks requested; enabling --bench-eval.")
    args.bench_eval = True
if args.early_stop_metric == "bench":
    print0("Warning: early stopping on benchmarks leaks test performance; forcing --early-stop-metric=val_bpb.")
    args.early_stop_metric = "val_bpb"
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
if not use_dummy_wandb and not hasattr(wandb, "init"):
    print0("wandb.init unavailable; disabling wandb logging.")
    use_dummy_wandb = True
if use_dummy_wandb:
    wandb_run = DummyWandb()
else:
    try:
        wandb_run = wandb.init(project="nanochat-mid", name=args.run, config=user_config)
    except Exception as exc:
        print0(f"wandb init failed ({exc}); disabling wandb logging.")
        wandb_run = DummyWandb()

# Load the model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and args.device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device-batch-size to this script?")
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=args.weight_decay)
adamw_optimizer, muon_optimizer = optimizers
# Override the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# Midtraining data mixture and DataLoader
base_dir = get_base_dir()
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)

extra_jsonl_tasks = [CustomJSON(filepath=fp) for fp in args.extra_jsonl]

HENDRYCKS_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Holdout sizes for dev/val splits drawn from train-only datasets.
# These are excluded from training to avoid leakage into early stopping.
SMOLTALK_DEV_SIZE = 24000
GSM8K_DEV_SIZE = 420
HENDRYCKS_DEV_SIZE = 100

SDPO_REASONING_DATASETS = [
    "gsm8k-platinum",
    "gsm8k-567",
    "hendrycks-math-benchmark",
    "math_qa",
    "calc-svamp",
    "math-augmented",
    "aqua-rat-mcqa",
    "logiqa",
    "social-iqa",
    "qasc",
]

class RepeatTask(Task):
    """Lightweight task wrapper to up-weight a dataset without reloading it."""
    def __init__(self, task, repeats=1, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.repeats = max(1, int(repeats))

    @property
    def eval_type(self):
        return self.task.eval_type

    def num_examples(self):
        return len(self.task) * self.repeats

    def get_example(self, index):
        if len(self.task) == 0:
            raise ValueError("RepeatTask has empty base task")
        return self.task[index % len(self.task)]

def _maybe_repeat(task, repeats):
    return RepeatTask(task, repeats) if repeats and int(repeats) > 1 else task

def _extra_repeats(task, repeats):
    repeats = int(repeats) if repeats is not None else 1
    if repeats <= 1:
        return []
    return [RepeatTask(task, repeats - 1)]

def _phase_weight(base_weight, phase2=False):
    mult = args.phase2_upweight_multiplier if phase2 else 1
    return max(1, int(base_weight) * int(mult))

def build_default_train_tasks():
    return [
        SmolTalk(split="train", start=SMOLTALK_DEV_SIZE),  # 460K rows of general conversations
        Alpaca(split="train"),  # instruction-following (helps IFEval-style constraints)
        MMLU(subset="auxiliary_train", split="train"),  # 100K rows of multiple choice problems
        GSM8K(subset="main", split="train", start=GSM8K_DEV_SIZE),  # 8K rows teaching simple math and (calculator) tool use
        CustomJSON(filepath=identity_conversations_filepath),  # 1K rows of synthetic identity conversations
        CustomJSON(filepath=identity_conversations_filepath),  # repeat for more weight
        SimpleSpelling(size=200000, split="train"),  # spelling practice
        SpellingBee(size=80000, split="train"),  # spelling + counting with Python verification style
    ]

def build_benchmarks_train_tasks(phase2=False):
    # Benchmark-oriented profile: add (train-split) datasets that cover reasoning, math, QA, and tool-call formatting.
    # NOTE: Some benchmarks are test-only; we intentionally avoid mixing test sets by default to reduce evaluation leakage.
    sdpo_reasoning_tasks, _ = build_sdpo_tasks(SDPO_REASONING_DATASETS, split="train", skip_errors=True)
    gpqa_tasks = []
    try:
        gpqa_task = GPQA(subset="gpqa_main", split="train")
        gpqa_tasks = [_maybe_repeat(gpqa_task, _phase_weight(args.upweight_gpqa, phase2))]
    except Exception as exc:
        print0(f"[mid_train] Skipping GPQA (train) due to load error: {exc}")
    xlam_tasks = []
    try:
        xlam_tasks = [XLAMFunctionCalling(split="train")]
    except Exception as exc:
        print0(f"[mid_train] Skipping XLAM-FC (train) due to load error: {exc}")
    extra_mmlu_subject_tasks = []
    try:
        subject_weights = {
            "physics": _phase_weight(args.upweight_mmlu_physics, phase2),
            "biology": _phase_weight(args.upweight_mmlu_biology, phase2),
            "engineering": _phase_weight(args.upweight_mmlu_engineering, phase2),
            "cs": _phase_weight(args.upweight_mmlu_cs, phase2),
            "it": _phase_weight(args.upweight_mmlu_it, phase2),
        }
        if args.mmlu_subject_upweight_split != "auxiliary_train":
            print0(
                f"[mid_train] Using MMLU split '{args.mmlu_subject_upweight_split}' for subject upweighting "
                "(may introduce evaluation leakage)."
            )
        subj_subset = "auxiliary_train" if args.mmlu_subject_upweight_split == "auxiliary_train" else "all"
        subj_split = "train" if args.mmlu_subject_upweight_split == "auxiliary_train" else args.mmlu_subject_upweight_split
        for subject_group, weight in subject_weights.items():
            if weight <= 1:
                continue
            subjects = MMLU_SUBJECT_GROUPS.get(subject_group, [])
            if not subjects:
                continue
            extra_mmlu_subject_tasks.extend(
                _extra_repeats(
                    MMLUSubjects(subjects=subjects, subset=subj_subset, split=subj_split),
                    weight,
                )
            )
    except Exception as exc:
        print0(f"[mid_train] Skipping MMLU subject upweighting due to load error: {exc}")
    return [
        # General chat + instruction following
        SmolTalk(split="train", start=SMOLTALK_DEV_SIZE, stop=SMOLTALK_DEV_SIZE + 200000),
        Alpaca(split="train"),
        # IFEval stays eval-only (train targets are empty in this codebase).
        # Reasoning / chain-of-thought style data
        OpenThoughts2(split="train", stop=200000),
        SkunkworksReasoning(split="train"),
        MagpieReasoning(split="train", stop=150000),
        NaturalReasoning(split="train", stop=50000),
        # Multiple choice + commonsense
        _maybe_repeat(MMLU(subset="auxiliary_train", split="train"), _phase_weight(args.upweight_mmlu, phase2)),
        *extra_mmlu_subject_tasks,
        # NOTE: MMLU-Pro has no train split; keep it eval-only to avoid validation/test leakage.
        _maybe_repeat(ARC(subset="ARC-Challenge", split="train"), _phase_weight(args.upweight_arc, phase2)),
        _maybe_repeat(ARC(subset="ARC-Easy", split="train"), _phase_weight(args.upweight_arc, phase2)),
        _maybe_repeat(HellaSwag(split="train"), _phase_weight(args.upweight_hellaswag, phase2)),
        SciQ(split="train"),
        OpenBookQA(split="train"),
        # Hard science QA (very small dataset; oversample by repetition)
        *gpqa_tasks,
        # Math
        _maybe_repeat(GSM8K(subset="main", split="train", start=GSM8K_DEV_SIZE), _phase_weight(args.upweight_gsm8k, phase2)),
        OpenMathInstruct2(split="train", stop=200000),
        NuminaMathQwQ(split="train", stop=200000),
        AIME2024(split="train"),
        AIME2025(split="train"),
        *[
            _maybe_repeat(
                HendrycksMath(subject=s, split="train", start=HENDRYCKS_DEV_SIZE),
                _phase_weight(args.upweight_hendrycks, phase2),
            )
            for s in HENDRYCKS_SUBJECTS
        ],
        *sdpo_reasoning_tasks,
        # Tool / function-call formatting (proxy for BFCL-style evaluations)
        *xlam_tasks,
        HermesFunctionCalling(subset="func-calling-singleturn", split="train"),
        HermesFunctionCalling(subset="glaive-function-calling-5k", split="train"),
        FunctionCallingShareGPT(split="train", stop=80000),
        # Factoid QA (proxy for SimpleQA-style evaluations). Cap to avoid swamping the mixture.
        _maybe_repeat(
            TriviaQA(subset="unfiltered", split="train", stop=50000),
            _phase_weight(args.upweight_triviaqa, phase2),
        ),
        # Identity + spelling/counted-token robustness
        CustomJSON(filepath=identity_conversations_filepath),
        CustomJSON(filepath=identity_conversations_filepath),
        SimpleSpelling(size=200000, split="train"),
        SpellingBee(size=80000, split="train"),
    ]

def build_default_val_tasks():
    return [
        SmolTalk(split="train", stop=SMOLTALK_DEV_SIZE),  # holdout from train
        MMLU(subset="all", split="dev"),  # dev split (avoid test leakage)
        GSM8K(subset="main", split="train", stop=GSM8K_DEV_SIZE),  # holdout from train
    ]

def build_benchmarks_val_tasks():
    # Prefer test splits; fall back to validation when test isn't available.
    return [
        # Multiple choice + commonsense
        ARC(subset="ARC-Challenge", split="test"),
        ARC(subset="ARC-Easy", split="test"),
        HellaSwag(split="validation"),
        MMLU(subset="all", split="test"),
        MMLUPro(split="test"),
        # Math
        GSM8K(subset="main", split="test"),
        *[HendrycksMath(subject=s, split="test") for s in HENDRYCKS_SUBJECTS],
        # Code
        HumanEval(),
        # Factoid QA
        TriviaQA(subset="unfiltered", split="validation"),
        # Long-form / mixed QA
        HLE(split="test"),
    ]

def build_benchmark_eval_tasks():
    def _try_splits(name, builders):
        last_exc = None
        for builder in builders:
            try:
                return builder()
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            print0(f"[mid_train] Skipping {name} eval due to load error: {last_exc}")
        return None

    eval_tasks = []

    # Only include benchmarks with explicit test splits.
    task = _try_splits(
        "ARC-Challenge",
        [lambda: ARC(subset="ARC-Challenge", split="test")],
    )
    if task is not None:
        eval_tasks.append(("ARC-Challenge", task))
    task = _try_splits(
        "ARC-Easy",
        [lambda: ARC(subset="ARC-Easy", split="test")],
    )
    if task is not None:
        eval_tasks.append(("ARC-Easy", task))

    task = _try_splits("MMLU", [lambda: MMLU(subset="all", split="test")])
    if task is not None:
        eval_tasks.append(("MMLU", task))

    task = _try_splits("MMLU-Pro", [lambda: MMLUPro(split="test")])
    if task is not None:
        eval_tasks.append(("MMLU-Pro", task))

    task = _try_splits("GSM8K", [lambda: GSM8K(subset="main", split="test")])
    if task is not None:
        eval_tasks.append(("GSM8K", task))

    task = _try_splits(
        "HendrycksMath",
        [lambda: TaskMixture([HendrycksMath(subject=s, split="test") for s in HENDRYCKS_SUBJECTS])],
    )
    if task is not None:
        eval_tasks.append(("HendrycksMath", task))

    task = _try_splits("HumanEval", [lambda: HumanEval()])
    if task is not None:
        eval_tasks.append(("HumanEval", task))

    task = _try_splits("HLE", [lambda: HLE(split="test")])
    if task is not None:
        eval_tasks.append(("HLE", task))

    try:
        eval_tasks.append(("BFCL-v3", build_bfcl_v3_benchmark()))
    except Exception as exc:
        print0(f"[mid_train] Skipping BFCL-v3 (eval) due to load error: {exc}")

    return eval_tasks

def _resolve_task_for_index(task_object, index):
    if hasattr(task_object, "index_map") and hasattr(task_object, "tasks"):
        task_idx, local_idx = task_object.index_map[index]
        return task_object.tasks[task_idx], local_idx
    if hasattr(task_object, "tasks") and hasattr(task_object, "lengths"):
        local_idx = index
        for task_idx, task_length in enumerate(task_object.lengths):
            if local_idx < task_length:
                return task_object.tasks[task_idx], local_idx
            local_idx -= task_length
    return task_object, None

def _evaluate_at_index(task_object, index, conversation, completion):
    subtask, _local_idx = _resolve_task_for_index(task_object, index)
    return subtask.evaluate(conversation, completion)

def run_generative_eval(task_object, tokenizer, model, engine, ddp, ddp_rank, ddp_world_size, num_samples,
                        max_new_tokens, temperature, top_k, max_problems=None):
    device = model.get_device()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [
            _evaluate_at_index(task_object, i, conversation, completion)
            for completion in completions
        ]
        passed = any(outcomes)
        total += 1
        num_passed += int(passed)

    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    return num_passed / total if total > 0 else 0.0

def run_categorical_eval(task_object, tokenizer, model, ddp, ddp_rank, ddp_world_size, batch_size, max_problems=None):
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache = {}
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            answer_pos = torch.tensor(answer_time_positions, dtype=torch.long, device=device)
            logits = model(prompt_ids, logits_positions=answer_pos)

        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            focus_logits = logits[idx, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            outcome = _evaluate_at_index(task_object, i0 + idx, conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    return num_passed / total if total > 0 else 0.0

def run_benchmark_eval(task_specs, tokenizer, model, engine, ddp, ddp_rank, ddp_world_size, max_problems,
                       batch_size, num_samples, max_new_tokens, temperature, top_k):
    results = {}
    for name, task_object in task_specs:
        print0(f"Benchmark eval: {name}")
        if task_object.eval_type == 'generative':
            acc = run_generative_eval(
                task_object,
                tokenizer,
                model,
                engine,
                ddp,
                ddp_rank,
                ddp_world_size,
                num_samples,
                max_new_tokens,
                temperature,
                top_k,
                max_problems=max_problems,
            )
        elif task_object.eval_type == 'categorical':
            acc = run_categorical_eval(
                task_object,
                tokenizer,
                model,
                ddp,
                ddp_rank,
                ddp_world_size,
                batch_size,
                max_problems=max_problems,
            )
        else:
            raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
        results[name] = acc
        print0(f"{name} accuracy: {100 * acc:.2f}%")
    return results

train_dataset_phase2 = None
train_epoch_size = None
use_phase2 = args.phase2_start_epoch > 0
if args.dataset_profile == "default":
    if use_phase2:
        print0("Warning: phase2 schedule is only supported with --dataset-profile=benchmarks; disabling phase2.")
        use_phase2 = False
    train_tasks = build_default_train_tasks()
    train_tasks.extend(extra_jsonl_tasks)
    train_dataset = TaskMixture(train_tasks)
    train_epoch_size = len(train_dataset)
elif args.dataset_profile == "benchmarks":
    train_tasks_phase1 = build_benchmarks_train_tasks(phase2=False)
    train_tasks_phase1.extend(extra_jsonl_tasks)
    train_dataset = TaskMixture(train_tasks_phase1)
    train_epoch_size = len(train_dataset)
    if use_phase2:
        train_tasks_phase2 = build_benchmarks_train_tasks(phase2=True)
        train_tasks_phase2.extend(extra_jsonl_tasks)
        train_dataset_phase2 = TaskMixture(train_tasks_phase2)
elif args.dataset_profile == "both":
    if use_phase2:
        print0("Warning: phase2 schedule is only supported with --dataset-profile=benchmarks; disabling phase2.")
        use_phase2 = False
    # Combine the two profiles for a single run.
    train_tasks = build_default_train_tasks() + build_benchmarks_train_tasks()
    train_tasks.extend(extra_jsonl_tasks)
    train_dataset = TaskMixture(train_tasks)
    train_epoch_size = len(train_dataset)
else:
    raise ValueError(f"Unknown --dataset-profile {args.dataset_profile}")

if args.eval_profile == "default":
    val_tasks = build_default_val_tasks()
elif args.eval_profile == "benchmarks":
    val_tasks = build_benchmarks_val_tasks()
elif args.eval_profile == "both":
    val_tasks = build_default_val_tasks() + build_benchmarks_val_tasks()
else:
    raise ValueError(f"Unknown --eval-profile {args.eval_profile}")
val_dataset = TaskMixture(val_tasks)
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the training dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of training
current_epoch = 1 # track epoch for logging
epochs_completed = 0 # track completed epochs for eval triggers
current_epoch_sync_for_loader = 1 # synced epoch gate for phase scheduling
def mid_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for midtraining with bestfit-crop packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm to minimize cropping.
    This matches the BOS-aligned approach used in pretraining.
    """
    global last_step, approx_progress, current_epoch, epochs_completed
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position

    # Conversation buffer: list of token lists
    conv_buffer = []
    cursor = ddp_rank  # Each rank processes different conversations (for fetching)
    consumed = ddp_rank  # Track actual consumption separately from buffering
    it = 0  # iteration counter
    active_phase = 1

    def _select_train_dataset(consumed_count):
        if not use_phase2:
            return train_dataset, 1
        epoch = current_epoch_sync_for_loader
        if epoch >= args.phase2_start_epoch:
            return train_dataset_phase2, 2
        return train_dataset, 1

    def refill_buffer():
        nonlocal cursor
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            conv_buffer.append(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                # Note: last_step is now triggered based on consumption, not fetching

    while True:
        if split == "train":
            selected_dataset, phase = _select_train_dataset(consumed)
            if phase != active_phase:
                conv_buffer.clear()
                cursor = ddp_rank
                active_phase = phase
            dataset = selected_dataset
            dataset_size = len(dataset)
        else:
            dataset = val_dataset
            dataset_size = len(dataset)
        rows = []
        for _ in range(args.device_batch_size):
            row = []
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, conv in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    consumed += ddp_world_size  # Track actual consumption
                else:
                    # No conversation fits - crop first conversation to fill remaining
                    conv = conv_buffer.pop(0)
                    row.extend(conv[:remaining])
                    consumed += ddp_world_size  # Track actual consumption

            rows.append(row[:row_capacity])

        # Stopping condition to respect num_iterations, if given
        it += 1
        if 0 < args.num_iterations <= it and split == "train":
            last_step = True

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        if split == "train":
            completed = consumed // train_epoch_size
            if completed > epochs_completed:
                epochs_completed = completed
            current_epoch = min(args.num_epochs, completed + 1)
            if args.num_iterations > 0:
                approx_progress = min(it / args.num_iterations, 1.0)
            else:
                total_target = train_epoch_size * args.num_epochs
                approx_progress = min(consumed / total_target, 1.0)
            # Trigger last_step when we've consumed enough (instead of when cursor wraps)
            if args.num_iterations < 0 and consumed >= train_epoch_size * args.num_epochs:
                last_step = True

        # Build tensors
        use_cuda = device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        yield inputs, targets

train_loader = mid_data_generator_bos_bestfit("train")
build_val_loader = lambda: mid_data_generator_bos_bestfit("val")
progress = 0 # will go from 0 to 1 over the course of training
base_replay_loader = None
base_replay_rng = random.Random(1234)
if args.base_replay_mix > 0.0:
    if args.base_replay_datasets:
        os.environ["NANOCHAT_BASE_DATASETS"] = args.base_replay_datasets
    try:
        base_replay_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            split="train",
            tokenizer_batch_size=args.base_replay_tokenizer_batch_size,
            device=device_type,
            buffer_size=args.base_replay_buffer_size,
        )
        print0(f"[mid_train] Base replay enabled: mix={args.base_replay_mix:.2f}")
        if args.num_iterations < 0:
            print0(
                "[mid_train] Warning: base replay is enabled with --num-epochs; "
                "total steps will increase because base batches are added on top "
                "of mid-train consumption. Use --num-iterations to keep steps fixed."
            )
    except Exception as exc:
        print0(f"[mid_train] Base replay disabled due to load error: {exc}")
        base_replay_loader = None
        args.base_replay_mix = 0.0

def _next_train_batch():
    if base_replay_loader is not None and base_replay_rng.random() < args.base_replay_mix:
        return next(base_replay_loader)
    return next(train_loader)

# Learning rate scheduler
def get_lr_multiplier(progress):
    # first 80% of training: no decay, then linearly ramp down to 0.
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

def build_checkpoint_meta(step, val_bpb, bench_score=None):
    meta = {
        "step": step,
        "val_bpb": val_bpb,
        "model_config": {
            "sequence_len": args.max_seq_len,
            "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": depth,
            "n_head": model.config.n_head,
            "n_kv_head": model.config.n_kv_head,
            "n_embd": model.config.n_embd,
        },
        "user_config": user_config,
    }
    if bench_score is not None:
        meta["bench_score"] = bench_score
    return meta

# -----------------------------------------------------------------------------
# Training loop
x, y = _next_train_batch() # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = 0
# If eval is epoch-based, trigger a baseline eval at step 0.
last_eval_epoch = -1 if args.eval_every_epoch else 0
val_bpb = None
early_stopper = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
bench_eval_tasks = None
bench_eval_engine = None
last_bench_score = None
best_bench_score = None
best_bench_step = None
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # Synchronize epoch progress so all ranks take the same eval path.
    epochs_completed_sync = epochs_completed
    if ddp:
        epoch_tensor = torch.tensor(epochs_completed, dtype=torch.int64, device=device)
        dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MIN)
        epochs_completed_sync = int(epoch_tensor.item())
    current_epoch_sync = min(args.num_epochs, epochs_completed_sync + 1)
    current_epoch_sync_for_loader = current_epoch_sync

    # once in a while: evaluate the val bpb (all ranks participate)
    eval_due = False
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        eval_due = True
    if args.eval_every_epoch and epochs_completed_sync > last_eval_epoch:
        eval_due = True
    if eval_due:
        model.eval()
        orig_model.eval()
        val_loader = build_val_loader()
        eval_steps = max(1, args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size))
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f} | epoch: {current_epoch_sync}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        early_stop_should_stop = False
        early_stop_best = None
        if args.early_stop_metric == "val_bpb":
            _improved, early_stop_should_stop = early_stopper.update(val_bpb)
            early_stop_best = early_stopper.best if early_stopper.best is not None else val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
            "val/epoch": current_epoch_sync,
            **(
                {
                    "val/early_stop_bad_evals": early_stopper.bad_evals,
                    "val/early_stop_best": early_stop_best,
                }
                if args.early_stop_metric == "val_bpb"
                else {}
            ),
        })
        if epochs_completed_sync > last_eval_epoch:
            last_eval_epoch = epochs_completed_sync
        if early_stop_should_stop and args.early_stop_metric == "val_bpb":
            print0(
                f"Early stopping triggered at step {step:05d} "
                f"(best bpb: {early_stop_best:.4f}, bad evals: {early_stopper.bad_evals})."
            )
            last_step = True
        if args.bench_eval:
            if bench_eval_tasks is None:
                bench_eval_tasks = build_benchmark_eval_tasks()
                bench_eval_engine = Engine(orig_model, tokenizer)
            with autocast_ctx:
                bench_results = run_benchmark_eval(
                    bench_eval_tasks,
                    tokenizer,
                    orig_model,
                    bench_eval_engine,
                    ddp,
                    ddp_rank,
                    ddp_world_size,
                    args.bench_eval_max_problems,
                    args.bench_eval_batch_size,
                    args.bench_eval_num_samples,
                    args.bench_eval_max_new_tokens,
                    args.bench_eval_temperature,
                    args.bench_eval_top_k,
                )
            if bench_results:
                last_bench_score = sum(bench_results.values()) / len(bench_results)
            else:
                last_bench_score = 0.0
            wandb_run.log({
                "step": step,
                "bench/score": last_bench_score,
                **{f"bench/{k}": v for k, v in bench_results.items()},
            })
            if args.early_stop_metric == "bench":
                if bench_results:
                    _improved, early_stop_should_stop = early_stopper.update(-last_bench_score)
                    early_stop_best = -early_stopper.best if early_stopper.best is not None else last_bench_score
                    wandb_run.log({
                        "step": step,
                        "bench/early_stop_bad_evals": early_stopper.bad_evals,
                        "bench/early_stop_best": early_stop_best,
                    })
                else:
                    print0("[mid_train] Benchmark results empty; skipping early stopping update.")
        if early_stop_should_stop and args.early_stop_metric == "bench":
            print0(
                f"Early stopping triggered at step {step:05d} "
                f"(best bench: {early_stop_best:.4f}, bad evals: {early_stopper.bad_evals})."
            )
            last_step = True
        model.train()
        orig_model.train()

    # save checkpoints periodically and at the end (only on master process)
    if master_process and not args.dry_run:
        should_save = last_step
        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            should_save = True
        if should_save:
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                [opt.state_dict() for opt in optimizers], # TODO: make sure saving across ranks is done correctly
                build_checkpoint_meta(step, val_bpb, bench_score=last_bench_score),
            )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y = _next_train_batch() # prefetch the next batch while the GPU is busy with forward/backward
        if args.num_iterations < 0:
            progress = max(progress, approx_progress) # only increase progress monotonically
    # step the optimizers
    if args.num_iterations > 0:
        progress = min((step + 1) / args.num_iterations, 1.0)
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1
    if args.num_iterations > 0 and step >= args.num_iterations:
        last_step = True

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch_sync} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch_sync,
        })

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not args.dry_run:
    from nanochat.report import get_report
    get_report().log(section="Midtraining", data=[
        user_config, # CLI args
        { # stats about the training setup
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        { # stats about training outcomes
            "Minimum validation bpb": min_val_bpb,
        }
    ])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
