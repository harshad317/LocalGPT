"""
Self-Distillation Policy Optimization (SDPO).

Implements SDPO from https://arxiv.org/abs/2601.20802:
- A feedback-conditioned self-teacher (same model, stop-grad) guides the student.
- The objective distills token-level distributions with KL/JS divergence.
- For RLVR (only scalar rewards), successful rollouts act as "correct solutions"
  to provide feedback for failed attempts in the same batch.

Run on one GPU:
python -m scripts.chat_sdpo

Or torchrun for training:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sdpo -- --run=sdpo
"""

import argparse
import itertools
import math
import os
import random
import re
from contextlib import nullcontext

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

try:
    import wandb
except Exception:
    wandb = None
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model, save_checkpoint 
from nanochat.engine import Engine
from tasks.sdpo_datasets import build_sdpo_tasks
from tasks.common import Task
from tasks.bfcl_v3 import build_bfcl_v3_benchmark
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.mmlu_pro import MMLUPro
from tasks.hle import HLE
from tasks.ifeval import IFEval
from tasks.xlam_function_calling import XLAMFunctionCalling

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Self-distillation policy optimization (SDPO)")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading (student)
parser.add_argument("--source", type=str, default="sft", help="base|mid|sft|rl - which checkpoint to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Teacher / reference model (defaults to the same as student)
parser.add_argument("--teacher-source", type=str, default=None, help="override teacher source (defaults to --source)")
parser.add_argument("--teacher-model-tag", type=str, default=None, help="override teacher model tag")
parser.add_argument("--teacher-step", type=int, default=None, help="override teacher step")
parser.add_argument("--teacher-refresh-every", type=int, default=-1, help="refresh teacher from student every N steps (-1 = never)")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over the training mixture")
parser.add_argument("--num-steps", type=int, default=-1, help="override number of steps (-1 = use num_epochs)")
# Dataset mix
parser.add_argument(
    "--train-datasets",
    type=str,
    default="all",
    help="comma-separated SDPO dataset keys or 'all'",
)
parser.add_argument(
    "--eval-dataset",
    type=str,
    default="gsm8k",
    help="dataset key for evaluation ('none' to disable)",
)
parser.add_argument(
    "--solution-source",
    type=str,
    default="auto",
    choices=["auto", "reference", "success"],
    help="how to select correct solutions for SDPO (auto=reference if available)",
)
# Anti-forgetting knobs (optional; preserve-knowledge defaults enable mild safeguards)
parser.add_argument(
    "--preserve-knowledge",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="enable conservative anti-forgetting defaults (replay + anchoring)",
)
parser.add_argument("--replay-mix", type=float, default=0.0, help="fraction of examples drawn from replay datasets (0 disables)")
parser.add_argument(
    "--replay-profile",
    type=str,
    default="none",
    choices=["none", "retain_benchmarks"],
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
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=4, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=8, help="number of samples per example/question (>=2)")
parser.add_argument("--score-batch-size", type=int, default=4, help="batch size for SDPO distillation forward passes")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# SDPO distillation
parser.add_argument("--distill-weight", type=float, default=1.0, help="scale for SDPO distillation loss")
parser.add_argument("--distill-loss", type=str, default="js", choices=["kl", "js"], help="divergence for SDPO (kl or generalized JS)")
parser.add_argument("--distill-alpha", type=float, default=0.5, help="alpha for generalized JS (0=fwd KL, 1=rev KL)")
parser.add_argument("--distill-top-k", type=int, default=128, help="top-k tokens for approximate distillation (0 = full vocab)")
parser.add_argument("--distill-add-tail", dest="distill_add_tail", action="store_true", help="add tail bucket for top-k distillation")
parser.add_argument("--no-distill-add-tail", dest="distill_add_tail", action="store_false", help="disable tail bucket for top-k distillation")
parser.set_defaults(distill_add_tail=False)
parser.add_argument("--distill-is-clip", type=float, default=None, help="IS clip for distillation ratio (None disables)")
parser.add_argument("--length-normalize", dest="length_normalize", action="store_true", help="average distillation loss per token (default)")
parser.add_argument("--no-length-normalize", dest="length_normalize", action="store_false", help="sum distillation loss over tokens")
parser.set_defaults(length_normalize=True)
# Reprompt templates / feedback
parser.add_argument("--max-reprompt-len", type=int, default=10240, help="max token length for reprompted prompt")
parser.add_argument("--reprompt-truncation", type=str, default="right", choices=["left", "right", "error"], help="reprompt truncation side")
parser.add_argument(
    "--reprompt-template",
    type=str,
    default="{prompt}{solution}{feedback}\n\nCorrectly solve the original question.",
    help="template for reprompting; placeholders: {prompt}, {solution}, {feedback}",
)
parser.add_argument(
    "--solution-template",
    type=str,
    default="\n\nCorrect solution:\n\n{successful_previous_attempt}",
    help="template for solution section; placeholder: {successful_previous_attempt}",
)
parser.add_argument(
    "--feedback-template",
    type=str,
    default="\n\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}",
    help="template for feedback section; placeholder: {feedback_raw}",
)
parser.add_argument("--dont-reprompt-on-self-success", dest="dont_reprompt_on_self_success", action="store_true", help="avoid using a sample's own success as demonstration (default)")
parser.add_argument("--reprompt-on-self-success", dest="dont_reprompt_on_self_success", action="store_false", help="allow using a sample's own success as demonstration")
parser.set_defaults(dont_reprompt_on_self_success=True)
parser.add_argument("--remove-thinking-from-demonstration", dest="remove_thinking_from_demonstration", action="store_true", help="strip <think>...</think> from demonstrations (default)")
parser.add_argument("--keep-thinking-in-demonstration", dest="remove_thinking_from_demonstration", action="store_false", help="keep <think>...</think> in demonstrations")
parser.set_defaults(remove_thinking_from_demonstration=True)
parser.add_argument("--include-environment-feedback", dest="include_environment_feedback", action="store_true", help="include environment feedback if available (default)")
parser.add_argument("--no-include-environment-feedback", dest="include_environment_feedback", action="store_false", help="disable environment feedback")
parser.set_defaults(include_environment_feedback=True)
parser.add_argument(
    "--environment-feedback-only-without-solution",
    dest="environment_feedback_only_without_solution",
    action="store_true",
    help="only use feedback when no solution is available (default)",
)
parser.add_argument(
    "--environment-feedback-always",
    dest="environment_feedback_only_without_solution",
    action="store_false",
    help="use feedback even when a solution exists",
)
parser.set_defaults(environment_feedback_only_without_solution=True)
# Regularized self-teacher (Appendix A.1)
parser.add_argument("--teacher-mix-alpha0", type=float, default=0.1, help="floor for mixing self-teacher with initial model; 0 disables")
parser.add_argument("--teacher-mix-schedule", type=str, default="linear", choices=["linear", "constant"], help="schedule for teacher mixing alpha")
parser.add_argument("--teacher-regularization", type=str, default="none", choices=["none", "ema", "trust-region"], help="teacher regularization mode")
parser.add_argument("--teacher-update-rate", type=float, default=0.05, help="EMA update rate or trust-region mix coefficient")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps (0 = disable)")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()

assert args.num_samples >= 2, "--num-samples must be >= 2 to use success-as-feedback in RLVR"

# -----------------------------------------------------------------------------
# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# Optional preserve-knowledge defaults
if args.preserve_knowledge:
    if args.replay_mix == 0.0:
        args.replay_mix = 0.2
    if args.replay_profile == "none":
        args.replay_profile = "retain_benchmarks"
    if args.anchor_l2 == 0.0:
        args.anchor_l2 = 1e-4
    if args.anchor_scope == "none":
        args.anchor_scope = "embeddings+lm_head"
    print0(
        "[chat_sdpo] preserve-knowledge enabled: "
        f"replay_mix={args.replay_mix}, replay_profile={args.replay_profile}, "
        f"anchor_l2={args.anchor_l2}, anchor_scope={args.anchor_scope}"
    )

user_config = vars(args).copy()

# wandb logging init
wandb_available = wandb is not None and hasattr(wandb, "init")
use_dummy_wandb = args.run == "dummy" or not master_process or not wandb_available
if not wandb_available and master_process and args.run != "dummy":
    print0("Warning: wandb is unavailable or missing init(); falling back to DummyWandb.")
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sdpo", name=args.run, config=user_config)

# Load the student model and tokenizer
model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)  # for sampling rollouts

# Self-teacher (defaults to the student model; stop-grad during distillation)
teacher_model = model
teacher_is_student = True
if args.teacher_source is not None or args.teacher_model_tag is not None or args.teacher_step is not None:
    teacher_source = args.teacher_source if args.teacher_source is not None else args.source
    teacher_tag = args.teacher_model_tag if args.teacher_model_tag is not None else args.model_tag
    teacher_step = args.teacher_step if args.teacher_step is not None else args.model_step
    teacher_model, _, _ = load_model(teacher_source, device, phase="eval", model_tag=teacher_tag, step=teacher_step)
    teacher_model.eval()
    teacher_is_student = False

teacher_regularization = args.teacher_regularization
teacher_update_rate = args.teacher_update_rate
if teacher_regularization in ("ema", "trust-region") and teacher_is_student:
    teacher_model, _, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
    teacher_model.eval()
    teacher_is_student = False

# Optional reference model for regularized self-teacher mixing (Appendix A.1)
ref_model = None
if args.teacher_mix_alpha0 > 0:
    ref_model, _, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.model_step)
    ref_model.eval()

# -----------------------------------------------------------------------------
# Optional parameter anchoring

anchor_params = []
if args.anchor_l2 > 0.0:
    if args.anchor_scope == "none":
        print0("[chat_sdpo] --anchor-l2 > 0 but --anchor-scope=none; disabling anchor.")
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
            print0("[chat_sdpo] Anchor scope produced no trainable params; disabling anchor.")
        else:
            print0(f"[chat_sdpo] Anchor enabled on {len(anchor_params)} param tensors (dtype={args.anchor_dtype}).")

# -----------------------------------------------------------------------------
# Replay helpers (anti-forgetting)

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


def build_replay_tasks(profile, cap):
    if profile == "none":
        return []
    if profile == "retain_benchmarks":
        print0("[chat_sdpo] retain_benchmarks replay uses some non-train splits (MMLU-Pro validation, HLE test).")
        tasks = []
        # MMLU (train split)
        try:
            tasks.append(("mmlu", _cap(MMLU(subset="auxiliary_train", split="train"), cap)))
        except Exception as exc:
            print0(f"[chat_sdpo] Skipping MMLU replay due to load error: {exc}")
        # MMLU-Pro has no train split; use validation (note: eval leakage risk)
        try:
            tasks.append(("mmlu-pro", _cap(MMLUPro(split="validation"), cap)))
        except Exception as exc:
            print0(f"[chat_sdpo] Skipping MMLU-Pro replay due to load error: {exc}")
        # GSM8K (train split)
        try:
            tasks.append(("gsm8k", _cap(GSM8K(subset="main", split="train"), cap)))
        except Exception as exc:
            print0(f"[chat_sdpo] Skipping GSM8K replay due to load error: {exc}")
        # XLAM function calling (train split)
        try:
            tasks.append(("xlam-fc", _cap(XLAMFunctionCalling(split="train"), cap)))
        except Exception as exc:
            print0(f"[chat_sdpo] Skipping XLAM-FC replay due to load error: {exc}")
        # IFEval (train split)
        try:
            tasks.append(("ifeval", _cap(IFEval(split="train"), cap)))
        except Exception as exc:
            print0(f"[chat_sdpo] Skipping IFEval replay due to load error: {exc}")
        # HLE is test-only (note: eval leakage risk)
        try:
            tasks.append(("hle", _cap(HLE(split="test"), cap)))
        except Exception as exc:
            print0(f"[chat_sdpo] Skipping HLE replay due to load error: {exc}")
        # BFCL-FC proxy (collection="python")
        try:
            tasks.append(("bfcl-fc", _cap(build_bfcl_v3_benchmark(collection="python"), cap)))
        except Exception as exc:
            print0(f"[chat_sdpo] Skipping BFCL-FC replay due to load error: {exc}")
        return tasks
    raise ValueError(f"Unknown replay profile: {profile}")

# -----------------------------------------------------------------------------
# Task data
train_tasks, train_dataset_names = build_sdpo_tasks(args.train_datasets, split="train")
if not train_tasks:
    raise ValueError("No training datasets resolved")

eval_task = None
if args.eval_dataset != "none":
    eval_tasks, _ = build_sdpo_tasks(args.eval_dataset, split="test")
    if eval_tasks:
        eval_task = eval_tasks[0]
    else:
        print0(f"Warning: eval dataset '{args.eval_dataset}' resolved to no tasks; disabling eval.")

train_index_map = []
for task_idx, task in enumerate(train_tasks):
    for local_idx in range(len(task)):
        train_index_map.append((task_idx, local_idx))
random.Random(42).shuffle(train_index_map)
train_num_examples = len(train_index_map)

replay_tasks = []
replay_dataset_names = []
replay_index_map = []
if not (0.0 <= args.replay_mix <= 1.0):
    raise ValueError("--replay-mix must be between 0.0 and 1.0")
if args.replay_mix > 0.0 and args.replay_profile == "none":
    print0("[chat_sdpo] Replay mix requested but --replay-profile=none; disabling replay.")
    args.replay_mix = 0.0
if args.replay_mix > 0.0 and args.replay_profile != "none":
    replay_task_specs = build_replay_tasks(args.replay_profile, args.replay_cap)
    if replay_task_specs:
        replay_dataset_names = [name for name, _ in replay_task_specs]
        replay_tasks = [task for _, task in replay_task_specs]
        for task_idx, task in enumerate(replay_tasks):
            for local_idx in range(len(task)):
                replay_index_map.append((task_idx, local_idx))
        random.Random(42).shuffle(replay_index_map)
        print0(f"[chat_sdpo] Replay enabled: {len(replay_index_map):,} rows, mix={args.replay_mix:.2f}")
    else:
        print0("[chat_sdpo] Replay requested but no tasks loaded; disabling replay.")
        args.replay_mix = 0.0

if args.num_steps > 0:
    num_steps = args.num_steps
else:
    num_steps = (train_num_examples // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")
print0(f"Training datasets: {', '.join(train_dataset_names)}")
if replay_dataset_names:
    print0(f"Replay datasets: {', '.join(replay_dataset_names)} (mix={args.replay_mix:.2f})")

# -----------------------------------------------------------------------------
# Helper functions

def _pad_batch(sequences, masks, pad_id):
    if len(sequences) == 0:
        raise ValueError("No sequences to pad")
    max_len = max(len(seq) for seq in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, (seq, m) in enumerate(zip(sequences, masks)):
        if len(seq) != len(m):
            raise ValueError(f"mask length {len(m)} != seq length {len(seq)}")
        n = len(seq)
        ids[i, :n] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, :n] = torch.tensor(m, dtype=torch.long, device=device)
    return ids, mask


def _content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content)
    return str(content)


def _get_reference_solution_text(conversation):
    messages = conversation["messages"]
    if not messages:
        return None
    last = messages[-1]
    if last.get("role") != "assistant":
        return None
    return _content_to_text(last.get("content", ""))


def _render_prompt_text(conversation):
    messages = conversation["messages"]
    if len(messages) < 1:
        return ""
    # Exclude the final assistant response (ground truth).
    prompt_messages = messages[:-1]
    parts = []
    for msg in prompt_messages:
        role = msg.get("role", "user")
        label = role.capitalize()
        parts.append(f"{label}: {_content_to_text(msg.get('content', ''))}")
    return "\n\n".join(parts).strip()


def _remove_thinking_trace(text):
    if not text:
        return text
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def _format_solution_section(solution_text):
    if not solution_text:
        return ""
    if not isinstance(solution_text, str):
        solution_text = str(solution_text)
    if args.remove_thinking_from_demonstration:
        solution_text = _remove_thinking_trace(solution_text)
    return args.solution_template.format(successful_previous_attempt=solution_text)


def _format_feedback_section(feedback_text):
    if not feedback_text:
        return ""
    if not isinstance(feedback_text, str):
        feedback_text = str(feedback_text)
    return args.feedback_template.format(feedback_raw=feedback_text)


def _compose_teacher_prompt(prompt_text, correct_solution=None, feedback=None):
    prompt_text = (prompt_text or "").strip()
    if correct_solution is None and feedback is None:
        return prompt_text
    solution_section = _format_solution_section(correct_solution) if correct_solution else ""
    feedback_section = _format_feedback_section(feedback) if feedback else ""
    reprompt = args.reprompt_template.format(
        prompt=prompt_text,
        solution=solution_section,
        feedback=feedback_section,
    )
    return reprompt.strip()


def _get_feedback(task, conversation, response_text):
    if not args.include_environment_feedback:
        return None
    for attr in ("feedback", "get_feedback"):
        fn = getattr(task, attr, None)
        if callable(fn):
            try:
                return fn(conversation, response_text)
            except Exception:
                return None
    if isinstance(conversation, dict):
        fb = conversation.get("feedback")
        if isinstance(fb, str) and fb.strip():
            return fb.strip()
    return None


def _truncate_prompt_tokens(ids, mask, max_tokens, truncation):
    if max_tokens is None or max_tokens <= 0:
        return ids, mask
    if len(ids) <= max_tokens:
        return ids, mask
    if truncation == "error":
        raise ValueError(f"Reprompt exceeds max length ({len(ids)} > {max_tokens})")
    if truncation == "left":
        return ids[-max_tokens:], mask[-max_tokens:]
    return ids[:max_tokens], mask[:max_tokens]


def build_teacher_tokens(
    tokenizer,
    prompt_text,
    response_tokens,
    correct_solution=None,
    feedback=None,
    max_prompt_tokens=None,
    truncation="right",
):
    user_content = _compose_teacher_prompt(prompt_text, correct_solution, feedback)
    conv = {"messages": [{"role": "user", "content": user_content}]}
    if max_prompt_tokens is None or max_prompt_tokens <= 0:
        render_max = 65536
    else:
        render_max = max_prompt_tokens if truncation == "right" else max(max_prompt_tokens * 4, 65536)
    ids, mask = tokenizer.render_conversation(conv, max_tokens=render_max)
    ids, mask = _truncate_prompt_tokens(ids, mask, max_prompt_tokens, truncation)
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    ids = ids + [assistant_start] + response_tokens
    mask = mask + [0] + [1] * len(response_tokens)
    return ids, mask


def _mix_logprobs(logp_a, logp_b, alpha):
    if alpha <= 0:
        return logp_a
    if alpha >= 1:
        return logp_b
    loga = math.log1p(-alpha)
    logb = math.log(alpha)
    return torch.logaddexp(logp_a + loga, logp_b + logb)


def _teacher_mix_alpha(step, total_steps, alpha0, schedule):
    if alpha0 <= 0:
        return 0.0
    if schedule == "constant":
        return alpha0
    # linear schedule: alpha_t = max(alpha0, 1 - t/T)
    denom = max(1, total_steps - 1)
    return max(alpha0, 1.0 - (step / denom))


def _update_ema_teacher(teacher_model, student_model, rate):
    if rate <= 0:
        return
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(1.0 - rate).add_(student_param.data, alpha=rate)


def sdpo_distill_loss(
    student_model,
    teacher_model,
    ref_model,
    student_sequences,
    student_masks,
    teacher_sequences,
    teacher_masks,
    pad_id,
    batch_size,
    distill_loss,
    distill_alpha,
    length_normalize,
    top_k,
    add_tail,
    is_clip,
    rollout_is_weights,
    teacher_mix_alpha,
    teacher_trust_region,
    use_autocast,
):
    if len(student_sequences) == 0:
        return None, None
    if batch_size <= 0:
        batch_size = len(student_sequences)

    total_loss = torch.zeros((), device=device)
    total_examples = torch.zeros((), device=device)
    total_tokens = torch.zeros((), device=device)

    for i in range(0, len(student_sequences), batch_size):
        batch_student_seq = student_sequences[i:i + batch_size]
        batch_student_mask = student_masks[i:i + batch_size]
        batch_teacher_seq = teacher_sequences[i:i + batch_size]
        batch_teacher_mask = teacher_masks[i:i + batch_size]

        student_ids, student_mask = _pad_batch(batch_student_seq, batch_student_mask, pad_id)
        teacher_ids, teacher_mask = _pad_batch(batch_teacher_seq, batch_teacher_mask, pad_id)

        student_inputs = student_ids[:, :-1]
        teacher_inputs = teacher_ids[:, :-1]
        student_target_mask = student_mask[:, 1:].bool()
        teacher_target_mask = teacher_mask[:, 1:].bool()

        student_counts = student_target_mask.sum(dim=1)
        teacher_counts = teacher_target_mask.sum(dim=1)
        if not torch.equal(student_counts, teacher_counts):
            raise ValueError("Teacher/student response token counts do not match")

        with use_autocast:
            student_logits = student_model(student_inputs)
        with torch.no_grad():
            with use_autocast:
                teacher_logits = teacher_model(teacher_inputs)
                ref_logits = None
                if ref_model is not None and teacher_mix_alpha > 0:
                    ref_logits = ref_model(teacher_inputs)

        student_logp_full = F.log_softmax(student_logits, dim=-1)
        teacher_logp_full = F.log_softmax(teacher_logits, dim=-1)
        if teacher_trust_region and teacher_trust_region > 0:
            teacher_logp_full = _mix_logprobs(teacher_logp_full, student_logp_full.detach(), teacher_trust_region)
        if ref_logits is not None:
            ref_logp = F.log_softmax(ref_logits, dim=-1)
            teacher_logp_full = _mix_logprobs(teacher_logp_full, ref_logp, teacher_mix_alpha)

        # Per-token log-probs for IS weighting
        target_ids = student_ids[:, 1:]
        student_token_logp = torch.gather(student_logp_full, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        old_token_logp = student_token_logp.detach()

        student_logp = student_logp_full[student_target_mask]
        teacher_logp = teacher_logp_full[teacher_target_mask]

        def _add_tail(logp):
            log_s = torch.logsumexp(logp, dim=-1, keepdim=True)
            log_s = torch.clamp(log_s, max=-1e-7)
            tail_log = torch.log(-torch.expm1(log_s))
            return torch.cat([logp, tail_log], dim=-1)

        def _renorm(logp):
            return logp - torch.logsumexp(logp, dim=-1, keepdim=True)

        def _generalized_js(student_logp_in, teacher_logp_in, alpha):
            alpha = float(alpha)
            if alpha <= 0.0:
                return (teacher_logp_in.exp() * (teacher_logp_in - student_logp_in)).sum(dim=-1)
            if alpha >= 1.0:
                return (student_logp_in.exp() * (student_logp_in - teacher_logp_in)).sum(dim=-1)
            mix_logp = torch.logsumexp(
                torch.stack(
                    [
                        student_logp_in + math.log(1.0 - alpha),
                        teacher_logp_in + math.log(alpha),
                    ],
                    dim=0,
                ),
                dim=0,
            )
            kl_t = F.kl_div(mix_logp, teacher_logp_in, reduction="none", log_target=True).sum(dim=-1)
            kl_s = F.kl_div(mix_logp, student_logp_in, reduction="none", log_target=True).sum(dim=-1)
            return torch.lerp(kl_s, kl_t, alpha)

        if top_k and top_k > 0:
            k = min(top_k, teacher_logp.size(-1))
            teacher_topk_logp, topk_idx = torch.topk(teacher_logp, k, dim=-1)
            student_topk_logp = torch.gather(student_logp, dim=-1, index=topk_idx)
            if add_tail:
                teacher_distill_logp = _add_tail(teacher_topk_logp)
                student_distill_logp = _add_tail(student_topk_logp)
            else:
                teacher_distill_logp = _renorm(teacher_topk_logp)
                student_distill_logp = _renorm(student_topk_logp)
            if distill_loss == "kl":
                token_loss = (teacher_distill_logp.exp() * (teacher_distill_logp - student_distill_logp)).sum(dim=-1)
            else:
                token_loss = _generalized_js(student_distill_logp, teacher_distill_logp, distill_alpha)
        else:
            if distill_loss == "kl":
                token_loss = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
            else:
                token_loss = _generalized_js(student_logp, teacher_logp, distill_alpha)

        if is_clip is not None:
            student_token_logp_flat = student_token_logp[student_target_mask]
            old_token_logp_flat = old_token_logp[student_target_mask]
            ratio = (student_token_logp_flat - old_token_logp_flat).clamp(min=-20.0, max=20.0).exp()
            ratio = ratio.clamp(max=is_clip)
            token_loss = token_loss * ratio

        if rollout_is_weights is not None:
            if rollout_is_weights.ndim == 2:
                token_loss = token_loss * rollout_is_weights[student_target_mask]
            else:
                token_loss = token_loss * rollout_is_weights

        # Aggregate per example for optional length normalization.
        offset = 0
        for count in student_counts.tolist():
            if count <= 0:
                continue
            loss_sum = token_loss[offset:offset + count].sum()
            if length_normalize:
                loss_sum = loss_sum / count
            total_loss = total_loss + loss_sum
            total_examples = total_examples + 1
            total_tokens = total_tokens + count
            offset += count

    if total_examples.item() == 0:
        return None, None
    mean_loss = total_loss / total_examples
    mean_len = total_tokens / total_examples
    return mean_loss, mean_len


@torch.no_grad()
def sample_candidates(prompt_tokens, num_samples, seed_base):
    generated_sequences = []
    generated_masks = []
    remaining = num_samples
    sampling_step = 0
    while remaining > 0:
        cur = min(args.device_batch_size, remaining)
        seed = hash((seed_base, sampling_step)) & 0x7FFFFFFF
        with autocast_ctx:
            seqs, masks = engine.generate_batch(
                prompt_tokens,
                num_samples=cur,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                seed=seed,
            )
        generated_sequences.extend(seqs)
        generated_masks.extend(masks)
        remaining -= cur
        sampling_step += 1
    return generated_sequences, generated_masks


@torch.no_grad()
def run_task_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """Evaluate task success@k and yield records."""
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= args.device_batch_size
        generated_token_sequences, _ = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            try:
                is_correct = task.evaluate(conversation, generated_text)
            except Exception:
                is_correct = False
            outcomes.append({
                "is_correct": is_correct
            })
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

# Learning rate scheduler
def get_lr_multiplier(it):
    return 1.0 - it / max(1, num_steps)

# DDP example sharding
assert args.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by ddp_world_size"
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

# Distillation weighting / teacher regularization knobs
distill_is_clip = args.distill_is_clip if args.distill_is_clip is not None and args.distill_is_clip > 0 else None
teacher_trust_region = teacher_update_rate if teacher_regularization == "trust-region" else 0.0
use_ema_teacher = teacher_regularization == "ema" and teacher_update_rate > 0

# -----------------------------------------------------------------------------
# Training loop

pad_token_id = tokenizer.encode_special("<|assistant_end|>")
rank_indices_main = range(ddp_rank, train_num_examples, ddp_world_size)
example_iter_main = itertools.cycle(rank_indices_main)
example_iter_replay = None
replay_rng = None
if replay_index_map and args.replay_mix > 0.0:
    rank_indices_replay = range(ddp_rank, len(replay_index_map), ddp_world_size)
    example_iter_replay = itertools.cycle(rank_indices_replay)
    replay_rng = random.Random(42 + ddp_rank)

for step in range(num_steps):
    last_step = step == num_steps - 1

    # Optional teacher refresh (only when using a separate teacher model)
    if teacher_regularization == "none" and not teacher_is_student and args.teacher_refresh_every > 0 and step > 0 and step % args.teacher_refresh_every == 0:
        teacher_model.load_state_dict(model.state_dict(), strict=True)
        teacher_model.eval()
        print0(f"Refreshed teacher at step {step}")

    # Periodic evaluation (success@k)
    if eval_task is not None and args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        successk = torch.zeros(args.device_batch_size, device=device)
        with autocast_ctx:
            records_iter = run_task_eval(
                eval_task,
                tokenizer,
                engine,
                num_samples=args.device_batch_size,
                max_examples=args.eval_examples,
                temperature=1.0,
                top_k=args.top_k if args.top_k > 0 else None
            )
            records = list(records_iter)
        for k in range(1, args.device_batch_size + 1):
            successk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(successk, op=dist.ReduceOp.SUM)
        successk = successk / max(1, num_records.item())
        print_successk = [f"Success@{k}: {successk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_successk)}")
        log_successk = {f"success@{k}": successk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_successk,
        })
        model.train()

    # SDPO updates
    model.train()
    mix_alpha = _teacher_mix_alpha(step, num_steps, args.teacher_mix_alpha0, args.teacher_mix_schedule)
    loss_sum = torch.zeros((), device=device)
    loss_count = torch.zeros((), device=device)
    len_sum = torch.zeros((), device=device)
    len_count = torch.zeros((), device=device)
    sdpo_pairs = torch.zeros((), device=device)
    success_count = torch.zeros((), device=device)
    success_denom = torch.zeros((), device=device)

    for example_step in range(examples_per_rank):
        use_replay = replay_rng is not None and replay_rng.random() < args.replay_mix
        if use_replay and example_iter_replay is not None:
            example_idx = next(example_iter_replay)
            task_idx, local_idx = replay_index_map[example_idx]
            task = replay_tasks[task_idx]
        else:
            example_idx = next(example_iter_main)
            task_idx, local_idx = train_index_map[example_idx]
            task = train_tasks[task_idx]
        conversation = task[local_idx]
        prompt_tokens = tokenizer.render_for_completion(conversation)
        prompt_text = _render_prompt_text(conversation)
        prefix_length = len(prompt_tokens)

        # Sample candidates from current policy
        generated_sequences, generated_masks = sample_candidates(
            prompt_tokens,
            args.num_samples,
            seed_base=(step, example_idx)
        )

        response_tokens_list = []
        response_text_list = []
        for seq in generated_sequences:
            response_tokens = seq[prefix_length:]
            response_tokens_list.append(response_tokens)
            response_text_list.append(tokenizer.decode(response_tokens))

        reference_solution = _get_reference_solution_text(conversation)
        use_reference = reference_solution is not None and args.solution_source in ("auto", "reference")
        feedback_list = [_get_feedback(task, conversation, resp) for resp in response_text_list]

        correct_flags = None
        eval_available = True
        if args.solution_source != "reference":
            flags = []
            for resp in response_text_list:
                try:
                    flags.append(bool(task.evaluate(conversation, resp)))
                except Exception:
                    eval_available = False
                    break
            if eval_available:
                correct_flags = flags
                success_count += sum(correct_flags)
                success_denom += len(correct_flags)

        success_indices = []
        if correct_flags:
            success_indices = [i for i, ok in enumerate(correct_flags) if ok]

        def _select_solution(idx):
            if use_reference:
                return reference_solution
            if not success_indices:
                return None
            if args.dont_reprompt_on_self_success and idx in success_indices:
                for j in success_indices:
                    if j != idx:
                        return response_text_list[j]
                return None
            return response_text_list[success_indices[0]]

        student_seqs = []
        student_masks = []
        teacher_seqs = []
        teacher_masks = []
        for idx, (seq, mask, resp_tokens, is_correct) in enumerate(
            zip(
                generated_sequences,
                generated_masks,
                response_tokens_list,
                correct_flags if correct_flags is not None else [False] * len(response_tokens_list),
            )
        ):
            if is_correct:
                continue
            solution_text = _select_solution(idx)
            feedback_text = feedback_list[idx] if feedback_list else None
            if args.environment_feedback_only_without_solution and solution_text:
                feedback_text = None
            if solution_text is None and (feedback_text is None or str(feedback_text).strip() == ""):
                continue
            teacher_ids, teacher_mask = build_teacher_tokens(
                tokenizer,
                prompt_text,
                resp_tokens,
                correct_solution=solution_text,
                feedback=feedback_text,
                max_prompt_tokens=args.max_reprompt_len,
                truncation=args.reprompt_truncation,
            )
            student_seqs.append(seq)
            student_masks.append(mask)
            teacher_seqs.append(teacher_ids)
            teacher_masks.append(teacher_mask)

        if not student_seqs:
            continue

        sdpo_loss, mean_len = sdpo_distill_loss(
            model,
            teacher_model,
            ref_model,
            student_seqs,
            student_masks,
            teacher_seqs,
            teacher_masks,
            pad_token_id,
            batch_size=args.score_batch_size,
            distill_loss=args.distill_loss,
            distill_alpha=args.distill_alpha,
            length_normalize=args.length_normalize,
            top_k=args.distill_top_k,
            add_tail=args.distill_add_tail,
            is_clip=distill_is_clip,
            rollout_is_weights=None,
            teacher_mix_alpha=mix_alpha,
            teacher_trust_region=teacher_trust_region,
            use_autocast=autocast_ctx,
        )
        if sdpo_loss is None:
            continue
        (sdpo_loss * args.distill_weight / examples_per_rank).backward()

        loss_sum += sdpo_loss.detach()
        loss_count += 1
        len_sum += mean_len.detach()
        len_count += 1
        sdpo_pairs += len(student_seqs)

    anchor_loss_item = None
    if anchor_params:
        anchor_loss = 0.0
        for _name, param, ref in anchor_params:
            anchor_loss = anchor_loss + (param - ref).pow(2).mean()
        (anchor_loss * args.anchor_l2).backward()
        anchor_loss_item = anchor_loss.detach()

    # Logging across ranks
    if ddp:
        for t in [loss_sum, loss_count, len_sum, len_count, sdpo_pairs, success_count, success_denom]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    mean_loss = (loss_sum / loss_count.clamp(min=1)).item()
    mean_len = (len_sum / len_count.clamp(min=1)).item()
    success_rate = (success_count / success_denom.clamp(min=1)).item() if success_denom.item() > 0 else 0.0
    anchor_str = ""
    if anchor_loss_item is not None:
        anchor_str = f" | anchor: {anchor_loss_item.item():.6f}"
    print0(
        f"Step {step}/{num_steps} | loss: {mean_loss:.6f} | "
        f"sdpo_pairs: {int(sdpo_pairs.item())} | success_rate: {success_rate:.3f} | "
        f"resp_tokens: {mean_len:.1f} | mix_alpha: {mix_alpha:.3f}{anchor_str}"
    )
    log_payload = {
        "step": step,
        "loss": mean_loss,
        "sdpo_pairs": int(sdpo_pairs.item()),
        "success_rate": success_rate,
        "resp_tokens": mean_len,
        "mix_alpha": mix_alpha,
    }
    if anchor_loss_item is not None:
        log_payload["anchor_loss"] = anchor_loss_item.item()
    wandb_run.log(log_payload)

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    if use_ema_teacher and teacher_model is not model:
        _update_ema_teacher(teacher_model, model, teacher_update_rate)
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Save checkpoints (skip step 0, save last step)
    if master_process and ((step > 0 and step % args.save_every == 0) or last_step):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatsdpo_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SDPO", data=[
    user_config,
    {
        "Number of iterations": num_steps,
        "Examples per step": args.examples_per_step,
        "Distill loss": args.distill_loss,
        "Distill alpha": args.distill_alpha,
        "Distill top-k": args.distill_top_k,
        "Distill add tail": args.distill_add_tail,
        "Distill IS clip": args.distill_is_clip,
        "Teacher mix alpha0": args.teacher_mix_alpha0,
        "Teacher regularization": args.teacher_regularization,
        "Teacher update rate": args.teacher_update_rate,
        "Reprompt max len": args.max_reprompt_len,
        "Reprompt truncation": args.reprompt_truncation,
        "Include env feedback": args.include_environment_feedback,
        "Env feedback only w/o solution": args.environment_feedback_only_without_solution,
        "Dont reprompt on self success": args.dont_reprompt_on_self_success,
        "Remove thinking from demo": args.remove_thinking_from_demonstration,
        "Train datasets": train_dataset_names,
        "Replay datasets": replay_dataset_names,
        "Replay mix": args.replay_mix,
        "Anchor L2": args.anchor_l2,
        "Anchor scope": args.anchor_scope,
    },
])

wandb_run.finish()
compute_cleanup()
