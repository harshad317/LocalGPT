"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import argparse
import os
import random
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from nanochat.early_stopping import EarlyStopping
from scripts.chat_eval import run_chat_eval

from tasks.aime import AIME2024, AIME2025
from tasks.arc import ARC
from tasks.common import Task, TaskMixture
from tasks.bfcl_v3 import build_bfcl_v3_benchmark
from tasks.gpqa import GPQA
from tasks.gsm8k import GSM8K
from tasks.hellaswag import HellaSwag
from tasks.hendrycks_math import HendrycksMath
from tasks.hle import HLE
from tasks.humaneval import HumanEval
from tasks.ifeval import IFEval
from tasks.mmlu import MMLU
from tasks.mmlu_pro import MMLUPro
from tasks.sdpo_datasets import build_sdpo_tasks
from tasks.triviaqa import TriviaQA
from tasks.xlam_function_calling import XLAMFunctionCalling

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised finetuning for chat")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="mid", help="base|mid - which checkpoint to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--num-iterations", type=int, default=-1, help="override number of iterations (-1 = use num_epochs)")
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=4, help="per-device batch size")
parser.add_argument("--target-examples-per-step", type=int, default=32, help="target examples per optimization step")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.02, help="initial LR as fraction of base LR")
# Anti-forgetting knobs (optional; preserve-knowledge defaults enable mild safeguards)
parser.add_argument(
    "--preserve-knowledge",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="enable conservative anti-forgetting defaults (replay + anchoring)",
)
parser.add_argument("--replay-mix", type=float, default=0.0, help="fraction of batches drawn from replay datasets (0 disables)")
parser.add_argument(
    "--replay-profile",
    type=str,
    default="none",
    choices=["none", "midtrain_min", "sdpo_reasoning", "retain_benchmarks"],
    help="which replay dataset mix to use when --replay-mix > 0",
)
parser.add_argument("--replay-cap", type=int, default=50_000, help="max examples per replay dataset")
parser.add_argument("--anchor-l2", type=float, default=0.0, help="L2 anchoring strength to initial params (0 disables)")
parser.add_argument(
    "--anchor-scope",
    type=str,
    default="none",
    choices=["none", "scalars", "embeddings", "lm_head", "embeddings+lm_head", "all"],
    help="which params to anchor when --anchor-l2 > 0",
)
parser.add_argument(
    "--anchor-dtype",
    type=str,
    default="bfloat16",
    choices=["bfloat16", "float32"],
    help="dtype for anchor parameter copies (smaller saves memory)",
)
parser.add_argument(
    "--freeze-embeddings",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="freeze token embeddings during SFT",
)
parser.add_argument(
    "--freeze-lm-head",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="freeze lm_head during SFT",
)
parser.add_argument(
    "--freeze-scalars",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="freeze resid/x0 scalars during SFT",
)
parser.add_argument(
    "--freeze-blocks",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="freeze all transformer blocks during SFT (requires optimizer to skip frozen params)",
)
parser.add_argument(
    "--freeze-first-n-blocks",
    type=int,
    default=0,
    help="freeze the first N transformer blocks during SFT",
)
# Evaluation
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val loss every N steps")
parser.add_argument("--eval-steps", type=int, default=100, help="number of batches for val loss evaluation")
parser.add_argument("--eval-metrics-every", type=int, default=200, help="evaluate accuracy metrics every N steps")
parser.add_argument("--eval-metrics-max-problems", type=int, default=1024, help="max problems per metric evaluation")
parser.add_argument(
    "--eval-every-epoch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="evaluate at the end of every epoch (overrides --eval-every/--eval-metrics-every)",
)
parser.add_argument(
    "--bench-eval",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="run benchmark task evals at each eval step",
)
parser.add_argument("--bench-eval-max-problems", type=int, default=None, help="max problems per benchmark (None = all)")
parser.add_argument("--bench-eval-batch-size", type=int, default=8, help="batch size for categorical benchmark evals")
parser.add_argument("--bench-eval-num-samples", type=int, default=1, help="num samples per benchmark problem")
parser.add_argument("--bench-eval-max-new-tokens", type=int, default=512, help="max tokens to generate per benchmark problem")
parser.add_argument("--bench-eval-temperature", type=float, default=0.0, help="temperature for benchmark generation")
parser.add_argument("--bench-eval-top-k", type=int, default=50, help="top-k sampling for benchmark generation")
parser.add_argument(
    "--early-stop-patience",
    type=int,
    default=1,
    help="early stop after N non-improving benchmark evals (0 disables)",
)
parser.add_argument("--early-stop-min-delta", type=float, default=0.0, help="min improvement in benchmark score to reset patience")
args = parser.parse_args()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Optional preserve-knowledge defaults
if args.preserve_knowledge:
    if args.replay_mix == 0.0:
        args.replay_mix = 0.2
    if args.replay_profile == "none":
        args.replay_profile = "midtrain_min"
    if args.anchor_l2 == 0.0:
        args.anchor_l2 = 1e-4
    if args.anchor_scope == "none":
        args.anchor_scope = "embeddings+lm_head"
    print0(
        "[chat_sft] preserve-knowledge enabled: "
        f"replay_mix={args.replay_mix}, replay_profile={args.replay_profile}, "
        f"anchor_l2={args.anchor_l2}, anchor_scope={args.anchor_scope}"
    )

user_config = vars(args).copy()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config, save_code=True)

# Load the model and tokenizer
model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
orig_model = model # original, uncompiled model
# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
engine = Engine(orig_model, tokenizer) # use the uncompiled model for evals

# -----------------------------------------------------------------------------
# Task data mixture we'll train on
# NOTE: This SFT mix intentionally excludes datasets already used in scripts/mid_train.py.
# We use alternative instruction/chat + code + function-calling datasets to improve
# generalization and transfer to the target benchmarks without reusing mid-train data.
# Optional anti-forgetting knobs are available via CLI (replay mix, parameter anchoring,
# and partial freezing of embeddings/lm_head/scalars).

class SliceTask(Task):
    """Lightweight wrapper to cap large datasets without reloading them."""
    def __init__(self, task, limit):
        super().__init__()
        self.task = task
        self.limit = min(int(limit), len(task))

    @property
    def eval_type(self):
        return self.task.eval_type

    def num_examples(self):
        return self.limit

    def get_example(self, index):
        return self.task[index]


def _cap(task, limit):
    return SliceTask(task, limit) if limit is not None else task


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


def build_replay_tasks(profile, cap):
    if profile == "none":
        return []
    if profile == "sdpo_reasoning":
        tasks, _ = build_sdpo_tasks(SDPO_REASONING_DATASETS, split="train", skip_errors=True)
        return [_cap(t, cap) for t in tasks]
    if profile == "midtrain_min":
        tasks = []
        try:
            tasks.append(_cap(MMLU(subset="auxiliary_train", split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping MMLU replay due to load error: {exc}")
        try:
            tasks.append(_cap(GSM8K(subset="main", split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping GSM8K replay due to load error: {exc}")
        try:
            tasks.append(_cap(ARC(subset="ARC-Challenge", split="train"), cap))
            tasks.append(_cap(ARC(subset="ARC-Easy", split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping ARC replay due to load error: {exc}")
        try:
            tasks.append(_cap(HellaSwag(split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping HellaSwag replay due to load error: {exc}")
        return tasks
    if profile == "retain_benchmarks":
        print0("[chat_sft] retain_benchmarks replay uses some non-train splits (MMLU-Pro validation, HLE test).")
        tasks = []
        # MMLU (train split)
        try:
            tasks.append(_cap(MMLU(subset="auxiliary_train", split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping MMLU replay due to load error: {exc}")
        # MMLU-Pro has no train split; use validation (note: eval leakage risk)
        try:
            tasks.append(_cap(MMLUPro(split="validation"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping MMLU-Pro replay due to load error: {exc}")
        # GSM8K (train split)
        try:
            tasks.append(_cap(GSM8K(subset="main", split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping GSM8K replay due to load error: {exc}")
        # XLAM function calling (train split)
        try:
            tasks.append(_cap(XLAMFunctionCalling(split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping XLAM-FC replay due to load error: {exc}")
        # IFEval (train split)
        try:
            tasks.append(_cap(IFEval(split="train"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping IFEval replay due to load error: {exc}")
        # HLE is test-only (note: eval leakage risk)
        try:
            tasks.append(_cap(HLE(split="test"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping HLE replay due to load error: {exc}")
        # BFCL-FC proxy (collection="python")
        try:
            tasks.append(_cap(build_bfcl_v3_benchmark(collection="python"), cap))
        except Exception as exc:
            print0(f"[chat_sft] Skipping BFCL-FC replay due to load error: {exc}")
        return tasks
    raise ValueError(f"Unknown replay profile: {profile}")


# General instruction/chat data (not used in mid_train.py)
general_datasets = [
    "openorca",
    "ultrachat-200k",
    "lmsys-chat-1m",
    "sharegpt-52k",
    "oasst1",
    "oasst1-h2oai",
    "dolly-15k",
    "json-mermaid",
]
general_tasks, _ = build_sdpo_tasks(general_datasets, split="train", skip_errors=True)
if not general_tasks:
    raise ValueError("No general chat datasets could be loaded for SFT training.")

# Cap huge datasets so the mix stays balanced.
general_tasks = [_cap(t, 50_000) for t in general_tasks]

# Function-calling proxy for XLAM-FC (not used in mid_train.py).
bfcl_task = _cap(build_bfcl_v3_benchmark(collection="python"), 50_000)

train_ds = TaskMixture([
    *general_tasks,
    bfcl_task,
])

replay_ds = None
if not (0.0 <= args.replay_mix <= 1.0):
    raise ValueError("--replay-mix must be between 0.0 and 1.0")
if args.replay_mix > 0.0 and args.replay_profile == "none":
    print0("[chat_sft] Replay mix requested but --replay-profile=none; disabling replay.")
    args.replay_mix = 0.0
if args.replay_mix > 0.0 and args.replay_profile != "none":
    replay_tasks = build_replay_tasks(args.replay_profile, args.replay_cap)
    if replay_tasks:
        replay_ds = TaskMixture(replay_tasks)
        print0(f"[chat_sft] Replay enabled: {len(replay_ds):,} rows, mix={args.replay_mix:.2f}")
    else:
        print0("[chat_sft] Replay requested but no tasks loaded; disabling replay.")
        args.replay_mix = 0.0

# Validation: small general-chat slice (avoid mid_train datasets).
val_tasks, _ = build_sdpo_tasks(["openorca"], split="test", skip_errors=True)
if val_tasks:
    val_ds = _cap(val_tasks[0], 2_000)
else:
    val_ds = _cap(general_tasks[0], 2_000)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # mask out targets where mask is 0
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device) # move to device
        targets = targets.to(device)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = args.device_batch_size * ddp_world_size
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert args.target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if args.num_iterations == -1:
    # derive num_iterations from num_epochs and the size of the dataset
    assert args.num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // args.target_examples_per_step) * args.num_epochs
else:
    num_iterations = args.num_iterations
train_loader_main = sft_data_generator(train_ds, batch_size=args.device_batch_size)
train_loader = train_loader_main
if replay_ds is not None and args.replay_mix > 0.0:
    train_loader_replay = sft_data_generator(replay_ds, batch_size=args.device_batch_size)
    replay_rng = random.Random(42 + ddp_rank)
    def _mixed_loader():
        while True:
            if replay_rng.random() < args.replay_mix:
                yield next(train_loader_replay)
            else:
                yield next(train_loader_main)
    train_loader = _mixed_loader()
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size)

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
    eval_tasks = [
        ("ARC-Challenge", ARC(subset="ARC-Challenge", split="test")),
        ("ARC-Easy", ARC(subset="ARC-Easy", split="test")),
        ("HellaSwag", HellaSwag(split="validation")),
        ("MMLU", MMLU(subset="all", split="test")),
        ("MMLU-Pro", MMLUPro(split="test")),
        ("GSM8K", GSM8K(subset="main", split="test")),
        ("HendrycksMath", TaskMixture([HendrycksMath(subject=s, split="test") for s in HENDRYCKS_SUBJECTS])),
        ("AIME-2024", AIME2024(split="train")),
        ("AIME-2025", AIME2025(split="train")),
        ("HumanEval", HumanEval()),
        ("TriviaQA", TriviaQA(subset="unfiltered", split="validation")),
        ("HLE", HLE(split="test")),
        ("IFEval", IFEval(split="train")),
    ]
    try:
        eval_tasks.insert(5, ("GPQA", GPQA(subset="gpqa_main", split="train")))
    except Exception as exc:
        print0(f"[chat_sft] Skipping GPQA (eval) due to load error: {exc}")
    try:
        eval_tasks.insert(9, ("XLAM-FC", XLAMFunctionCalling(split="train")))
    except Exception as exc:
        print0(f"[chat_sft] Skipping XLAM-FC (eval) due to load error: {exc}")
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

# -----------------------------------------------------------------------------
# Optional freezing + parameter anchoring

if args.freeze_embeddings:
    model.transformer.wte.weight.requires_grad = False
if args.freeze_lm_head:
    model.lm_head.weight.requires_grad = False
if args.freeze_scalars:
    model.resid_lambdas.requires_grad = False
    model.x0_lambdas.requires_grad = False
if args.freeze_blocks and args.freeze_first_n_blocks > 0:
    raise ValueError("--freeze-blocks and --freeze-first-n-blocks are mutually exclusive")
if args.freeze_blocks:
    for block in model.transformer.h:
        for param in block.parameters():
            param.requires_grad = False
    print0(f"[chat_sft] Frozen all {len(model.transformer.h)} transformer blocks.")
elif args.freeze_first_n_blocks > 0:
    if args.freeze_first_n_blocks < 0:
        raise ValueError("--freeze-first-n-blocks must be >= 0")
    n_freeze = min(args.freeze_first_n_blocks, len(model.transformer.h))
    for block in model.transformer.h[:n_freeze]:
        for param in block.parameters():
            param.requires_grad = False
    print0(f"[chat_sft] Frozen first {n_freeze}/{len(model.transformer.h)} transformer blocks.")

anchor_params = []
if args.anchor_l2 > 0.0:
    if args.anchor_scope == "none":
        print0("[chat_sft] --anchor-l2 > 0 but --anchor-scope=none; disabling anchor.")
    else:
        anchor_dtype = torch.bfloat16 if args.anchor_dtype == "bfloat16" else torch.float32
        def _maybe_anchor(name, param):
            if param.requires_grad:
                anchor_params.append((name, param, param.detach().clone().to(dtype=anchor_dtype)))
        if args.anchor_scope in {"scalars", "all"}:
            _maybe_anchor("resid_lambdas", model.resid_lambdas)
            _maybe_anchor("x0_lambdas", model.x0_lambdas)
        if args.anchor_scope in {"embeddings", "embeddings+lm_head", "all"}:
            _maybe_anchor("wte", model.transformer.wte.weight)
        if args.anchor_scope in {"lm_head", "embeddings+lm_head", "all"}:
            _maybe_anchor("lm_head", model.lm_head.weight)
        if not anchor_params:
            print0("[chat_sft] Anchor scope produced no trainable params; disabling anchor.")
        else:
            print0(f"[chat_sft] Anchor enabled on {len(anchor_params)} param tensors (dtype={args.anchor_dtype}).")

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Eval schedule + early stopping
steps_per_epoch = max(1, len(train_ds) // args.target_examples_per_step)
if args.num_iterations != -1 and args.num_epochs > 0:
    steps_per_epoch = max(1, num_iterations // args.num_epochs)
eval_every_steps = steps_per_epoch if args.eval_every_epoch else args.eval_every
eval_metrics_every_steps = steps_per_epoch if args.eval_every_epoch else args.eval_metrics_every
if args.eval_every_epoch:
    print0(f"Eval schedule: every epoch ({steps_per_epoch} steps)")

early_stopper = None
if args.early_stop_patience > 0:
    if not args.bench_eval:
        print0("Warning: early stopping requested but --bench-eval is disabled; disabling early stopping.")
    else:
        early_stopper = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)

# Go!
step = 0
metrics = {}
bench_eval_tasks = None
bench_eval_engine = None
best_bench_score = None
best_bench_step = None
last_bench_score = None
train_loss_item = float("nan")
num_tokens_item = 0
val_loss = None
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    eval_due = eval_every_steps > 0 and (last_step or step % eval_every_steps == 0)
    metrics_due = eval_metrics_every_steps > 0 and (last_step or (step > 0 and step % eval_metrics_every_steps == 0))
    bench_due = args.bench_eval and eval_metrics_every_steps > 0 and (last_step or step % eval_metrics_every_steps == 0)

    if eval_due or metrics_due or bench_due:
        model.eval()
        orig_model.eval()

    # evaluate the validation loss
    if eval_due:
        val_loader = build_val_loader()
        losses = []
        for _ in range(args.eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })

    # evaluate accuracy of the multiple choice tasks (which are quick to run)
    if metrics_due:
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            metrics["mmlu_acc"] = run_chat_eval(
                "MMLU",
                orig_model,
                tokenizer,
                engine,
                batch_size=args.device_batch_size * 2,
                max_problems=args.eval_metrics_max_problems,
            )
            metrics["arc_easy_acc"] = run_chat_eval(
                "ARC-Easy",
                orig_model,
                tokenizer,
                engine,
                batch_size=args.device_batch_size * 2,
                max_problems=args.eval_metrics_max_problems,
            )
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })

    if bench_due:
        if bench_eval_tasks is None:
            bench_eval_tasks = build_benchmark_eval_tasks()
            bench_eval_engine = Engine(orig_model, tokenizer)
        with torch.no_grad(), autocast_ctx:
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
        if early_stopper is not None:
            if bench_results:
                _improved, early_stop_should_stop = early_stopper.update(-last_bench_score)
                early_stop_best = -early_stopper.best if early_stopper.best is not None else last_bench_score
                wandb_run.log({
                    "step": step,
                    "bench/early_stop_bad_evals": early_stopper.bad_evals,
                    "bench/early_stop_best": early_stop_best,
                })
                if early_stop_should_stop:
                    print0(
                        f"Early stopping triggered at step {step:05d} "
                        f"(best bench: {early_stop_best:.4f}, bad evals: {early_stopper.bad_evals})."
                    )
                    last_step = True
            else:
                print0("[chat_sft] Benchmark results empty; skipping early stopping update.")

    if eval_due or metrics_due or bench_due:
        model.train()
        orig_model.train()

    if last_step:
        break

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    anchor_loss_item = None
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
            anchor_loss = None
            if anchor_params:
                anchor_loss = 0.0
                for _name, param, ref in anchor_params:
                    anchor_loss = anchor_loss + (param - ref).pow(2).mean()
                loss = loss + args.anchor_l2 * anchor_loss
        train_loss = loss.detach() # for logging
        if anchor_loss is not None:
            anchor_loss_item = anchor_loss.detach().item()
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    log_payload = {
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    }
    if anchor_loss_item is not None:
        log_payload["train/anchor_loss"] = anchor_loss_item
    wandb_run.log(log_payload)
    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
    model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
    checkpoint_meta = {
        "step": step,
        "val_loss": val_loss,
        **metrics,
        "model_config": model_config_kwargs,
    }
    if last_bench_score is not None:
        checkpoint_meta["bench_score"] = last_bench_score
    if best_bench_score is not None:
        checkpoint_meta["best_bench_score"] = best_bench_score
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # note: we don't bother to save the optimizer state
        checkpoint_meta,
    )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Replay rows": len(replay_ds) if replay_ds is not None else 0,
        "Replay mix": args.replay_mix,
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
        **({"Benchmark score": last_bench_score} if last_bench_score is not None else {}),
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
