"""
Train model. From root directory of the project, run as:

python -m scripts.base_train.py

or distributed as:

torchrun --nproc_per_node=<NUM_GPUS> -m scripts.base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from contextlib import nullcontext

try:
    import wandb
except Exception:
    wandb = None
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
try:
    from nanochat.common import (
        compute_init,
        compute_cleanup,
        print0,
        DummyWandb,
        print_banner,
        get_base_dir,
        autodetect_device_type,
    )
except ImportError:
    # Back-compat: older `nanochat.common` may not define compute_init yet.
    import os
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

    print_banner = _maybe_attr("print_banner")
    if print_banner is None:
        def print_banner():
            return

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
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.early_stopping import EarlyStopping
from scripts.base_eval import evaluate_model
print_banner()

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
from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.hellaswag import HellaSwag
from tasks.hendrycks_math import HendrycksMath
from tasks.mbpp import MBPP
from tasks.triviaqa import TriviaQA
from tasks.humaneval import HumanEval
from tasks.mmlu_pro import MMLUPro
from tasks.hle import HLE
from tasks.ifeval import IFEval

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="base_train", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=16, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=32, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=64, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=4096, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=int, default=8, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument(
    "--device-batch-size",
    "--device_batch_size",
    dest="device_batch_size",
    type=int,
    default=32,
    help="per-device batch size",
)
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--bench-eval", action="store_true", help="run benchmark task evals at each val eval step")
parser.add_argument("--bench-eval-max-problems", type=int, default=None, help="max problems per benchmark (None = all)")
parser.add_argument("--bench-eval-batch-size", type=int, default=8, help="batch size for categorical benchmark evals")
parser.add_argument("--bench-eval-num-samples", type=int, default=1, help="num samples per benchmark problem")
parser.add_argument("--bench-eval-max-new-tokens", type=int, default=512, help="max tokens to generate per benchmark problem")
parser.add_argument("--bench-eval-temperature", type=float, default=0.0, help="temperature for benchmark generation")
parser.add_argument("--bench-eval-top-k", type=int, default=50, help="top-k sampling for benchmark generation")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Early stopping (based on validation bpb; requires --eval-every > 0)
parser.add_argument("--early-stopping-patience", type=int, default=2, help="stop after this many non-improving val evals (0 = disable)")
parser.add_argument("--early-stopping-min-delta", type=float, default=0.05, help="required absolute improvement in val bpb to reset patience")
parser.add_argument("--early-stopping-warmup-steps", type=int, default=0, help="ignore early stopping until this global step")
parser.add_argument("--early-stopping-metric", type=str, default="val_bpb", choices=["val_bpb", "bench_score"], help="metric for early stopping")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
if args.bench_eval and args.eval_every <= 0:
    print0("Warning: --bench-eval requires --eval-every > 0 to run benchmark evals.")
if args.early_stopping_metric == "bench_score" and not args.bench_eval:
    print0("Warning: --early-stopping-metric=bench_score requires --bench-eval; disabling early stopping.")
    args.early_stopping_patience = 0
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
wandb_available = wandb is not None and hasattr(wandb, "init")
use_dummy_wandb = args.run == "dummy" or not master_process or not wandb_available
if not wandb_available and master_process and args.run != "dummy":
    print0("Warning: wandb is unavailable or missing init(); falling back to DummyWandb.")
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = args.depth
model_dim = args.depth * args.aspect_ratio
def find_num_heads(model_dim, target_head_dim):
    # Find num_heads that divides model_dim evenly, with head_dim closest to target.
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1
num_heads = find_num_heads(model_dim, args.head_dim)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
batch_lr_scale = 1.0
reference_batch_size = 2**19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard
    # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} (reference: {reference_batch_size:,})")

# Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
weight_decay_scaled = args.weight_decay * (12 / args.depth)**2
if args.depth != 12:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = dict(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim, window_pattern=args.window_pattern)
with torch.device("meta"):
    # All tensors are created as meta tensors (they have shape/dtype but no data)
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device) # All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
num_scaling_params = orig_model.num_scaling_params()
print0(f"Number of parameters: {num_params:,} (scaling: {num_scaling_params:,})")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio (use scaling params per Kaplan et al.)
    target_tokens = args.target_param_data_ratio * num_scaling_params
    num_iterations = target_tokens // args.total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = args.total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {args.total_batch_size * num_iterations / num_scaling_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    adam_betas=adam_betas,
    scalar_lr=args.scalar_lr * batch_lr_scale,
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for Muon optimizer (linear to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Benchmark evaluation helpers
HENDRYCKS_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

def build_benchmark_eval_tasks():
    return [
        ("ARC-Challenge", ARC(subset="ARC-Challenge", split="test")),
        ("ARC-Easy", ARC(subset="ARC-Easy", split="test")),
        ("HellaSwag", HellaSwag(split="validation")),
        ("MMLU", MMLU(subset="all", split="test")),
        ("MMLU-Pro", MMLUPro(split="test")),
        ("GSM8K", GSM8K(subset="main", split="test")),
        ("HendrycksMath", TaskMixture([HendrycksMath(subject=s, split="test") for s in HENDRYCKS_SUBJECTS])),
        ("MBPP", MBPP(split="test")),
        ("HumanEval", HumanEval()),
        ("TriviaQA", TriviaQA(subset="unfiltered", split="validation")),
        ("HLE", HLE(split="test")),
        ("IFEval", IFEval(split="train")),
    ]

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

def build_checkpoint_meta(step, val_bpb, dataloader_state_dict, loop_state):
    return {
        "step": step,
        "val_bpb": val_bpb,
        "model_config": model_config_kwargs,
        "user_config": user_config,
        "device_batch_size": args.device_batch_size,
        "max_seq_len": args.max_seq_len,
        "dataloader_state_dict": dataloader_state_dict,
        "loop_state": loop_state,
    }

def build_loop_state(min_val_bpb, smooth_train_loss, total_training_time, early_stopper,
                     best_bench_score, best_bench_step, last_bench_score):
    return {
        "min_val_bpb": min_val_bpb,
        "smooth_train_loss": smooth_train_loss,
        "total_training_time": total_training_time,
        "early_stopping": None if early_stopper is None else early_stopper.state_dict(),
        "best_bench_score": best_bench_score,
        "best_bench_step": best_bench_step,
        "last_bench_score": last_bench_score,
    }

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
    early_stopping_state = None
    best_bench_score = None
    best_bench_step = None
    last_bench_score = None
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]
    early_stopping_state = loop_state.get("early_stopping", None)
    best_bench_score = loop_state.get("best_bench_score", None)
    best_bench_step = loop_state.get("best_bench_step", None)
    last_bench_score = loop_state.get("last_bench_score", None)

early_stopper = None
if args.early_stopping_patience > 0:
    assert args.eval_every > 0, "--early-stopping-patience requires --eval-every > 0"
    early_stopper = EarlyStopping(patience=args.early_stopping_patience, min_delta=args.early_stopping_min_delta)
    early_stopper.load_state_dict(early_stopping_state)
    print0(
        f"Early stopping enabled: patience={args.early_stopping_patience}, "
        f"min_delta={args.early_stopping_min_delta}, warmup_steps={args.early_stopping_warmup_steps}, "
        f"metric={args.early_stopping_metric}"
    )

bench_eval_tasks = None
bench_eval_engine = None

# -----------------------------------------------------------------------------
# Training loop
while True:
    stop_requested = False
    stop_reason = None
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or (step > 0 and step % args.eval_every == 0)):
        model.eval()
        orig_model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        if early_stopper is not None and step >= args.early_stopping_warmup_steps:
            if args.early_stopping_metric == "val_bpb":
                improved, should_stop = early_stopper.update(val_bpb)
                if improved:
                    print0(f"Early stopping: new best val bpb {early_stopper.best:.6f}")
                else:
                    remaining = max(0, early_stopper.patience - early_stopper.bad_evals)
                    print0(
                        f"Early stopping: no improvement ({early_stopper.bad_evals}/{early_stopper.patience}); "
                        f"{remaining} eval(s) remaining"
                    )
                if should_stop:
                    stop_requested = True
                    stop_reason = (
                        f"early stopping (best={early_stopper.best:.6f}, "
                        f"bad_evals={early_stopper.bad_evals}, patience={early_stopper.patience})"
                    )
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
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
            if best_bench_score is None or last_bench_score > best_bench_score:
                best_bench_score = last_bench_score
                best_bench_step = step
                print0(f"New best benchmark score: {best_bench_score:.4f} at step {best_bench_step:05d}")
                loop_state = build_loop_state(
                    min_val_bpb,
                    smooth_train_loss,
                    total_training_time,
                    early_stopper,
                    best_bench_score,
                    best_bench_step,
                    last_bench_score,
                )
                save_checkpoint(
                    checkpoint_dir,
                    step,
                    orig_model.state_dict(),
                    [opt.state_dict() for opt in optimizers],
                    build_checkpoint_meta(step, val_bpb, dataloader_state_dict, loop_state),
                    rank=ddp_rank,
                )
            if early_stopper is not None and step >= args.early_stopping_warmup_steps and args.early_stopping_metric == "bench_score":
                bench_value = -last_bench_score
                improved, should_stop = early_stopper.update(bench_value)
                if improved:
                    print0(f"Early stopping: new best bench score {last_bench_score:.6f}")
                else:
                    remaining = max(0, early_stopper.patience - early_stopper.bad_evals)
                    print0(
                        f"Early stopping: no improvement ({early_stopper.bad_evals}/{early_stopper.patience}); "
                        f"{remaining} eval(s) remaining"
                    )
                if should_stop:
                    stop_requested = True
                    stop_reason = (
                        f"early stopping (best={-early_stopper.best:.6f}, "
                        f"bad_evals={early_stopper.bad_evals}, patience={early_stopper.patience})"
                    )
        orig_model.train()
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
            "The largest mammal in the world is",
            "The fastest land animal is",
            "The longest river in the world is",
            "The tallest mountain in the world is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or stop_requested or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        loop_state = build_loop_state(
            min_val_bpb,
            smooth_train_loss,
            total_training_time,
            early_stopper,
            best_bench_score,
            best_bench_step,
            last_bench_score,
        )
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            build_checkpoint_meta(step, val_bpb, dataloader_state_dict, loop_state),
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step or stop_requested:
        if stop_reason is not None:
            print0(f"Stopping due to {stop_reason}.")
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
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": args.total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "Best benchmark score": best_bench_score,
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
