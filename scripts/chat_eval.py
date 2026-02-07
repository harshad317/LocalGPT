"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import argparse
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist

try:
    from nanochat.common import (
        compute_init,
        compute_cleanup,
        get_dist_info,
        print0,
        autodetect_device_type,
    )
except ImportError:
    # Back-compat: some checkouts/environments may have an older `nanochat.common`
    # that doesn't define `compute_init` yet. Provide minimal fallbacks so this
    # entrypoint can still run.
    import os

    import nanochat.common as _common

    def print0(s="", **kwargs):
        ddp_rank = int(os.environ.get("RANK", 0))
        if ddp_rank == 0:
            print(s, **kwargs)

    def get_dist_info():
        if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
            return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        return False, 0, 0, 1

    def autodetect_device_type():
        if torch.cuda.is_available():
            device_type = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
        print0(f"Autodetected device type: {device_type}")
        return device_type

    compute_init = getattr(_common, "compute_init", None)
    compute_cleanup = getattr(_common, "compute_cleanup", None)

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

            is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
            if is_ddp_requested and device_type == "cuda":
                torch.cuda.set_device(ddp_local_rank)
                device = torch.device("cuda", ddp_local_rank)
                dist.init_process_group(backend="nccl")
                dist.barrier()
            else:
                device = torch.device(device_type)

            return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

    if compute_cleanup is None:
        def compute_cleanup():
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()

# -----------------------------------------------------------------------------
# Back-compat for older `tasks.common` APIs.
# Some forks/checkouts may not export `Task` (or related helpers) from
# `tasks.common`, but the task modules import them at import-time.
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
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.mmlu_pro import MMLUPro
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee
from tasks.hellaswag import HellaSwag
from tasks.gpqa import GPQA
from tasks.hendrycks_math import HendrycksMath
from tasks.hle import HLE
from tasks.mbpp import MBPP
from tasks.triviaqa import TriviaQA
from tasks.xlam_function_calling import XLAMFunctionCalling
from tasks.bfcl_v3 import build_bfcl_v3_benchmark
from tasks.common import TaskMixture

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

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

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [
            _evaluate_at_index(task_object, i, conversation, completion)
            for completion in completions
        ]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # Return the accuracy
    return num_passed/total

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits at the per-example answer position.
        # Important: computing full (B, T, V) logits can OOM for large vocabularies.
        with torch.no_grad():
            answer_pos = torch.tensor(answer_time_positions, dtype=torch.long, device=device)
            logits = model(prompt_ids, logits_positions=answer_pos) # (B, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the available letters of the answer
            focus_logits = logits[idx, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = _evaluate_at_index(task_object, i0 + idx, conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None):
    # Create the evaluation object
    task_name = task_name.strip()
    task_registry = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'MMLU-Pro': partial(MMLUPro, split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
        'HellaSwag': partial(HellaSwag, split="validation"),
        # NOTE: HF GPQA currently exposes only a train split; treat this as a proxy score.
        'GPQA': partial(GPQA, subset="gpqa_diamond", split="train"),
        # Math proxy for AIME-style problems.
        'HendrycksMath': lambda: TaskMixture([
            HendrycksMath(subject="algebra", split="test"),
            HendrycksMath(subject="counting_and_probability", split="test"),
            HendrycksMath(subject="geometry", split="test"),
            HendrycksMath(subject="intermediate_algebra", split="test"),
            HendrycksMath(subject="number_theory", split="test"),
            HendrycksMath(subject="prealgebra", split="test"),
            HendrycksMath(subject="precalculus", split="test"),
        ]),
        # Code proxy for HumanEval-like skills.
        'MBPP': partial(MBPP, split="test"),
        # Factoid QA proxy for SimpleQA-like evaluation.
        'TriviaQA': partial(TriviaQA, subset="unfiltered", split="validation"),
        # Function calling proxy for BFCL-style formatting (JSON validity pass-rate).
        'XLAM-FC': partial(XLAMFunctionCalling, split="train", stop=2000),
        # BFCL v3 benchmark (collection of function calling subsets).
        'BFCL-v3': lambda: build_bfcl_v3_benchmark(),
        # Humanity's Last Exam (multidomain, mix of exact-match + multiple choice).
        'HLE': partial(HLE, split="test"),
    }
    if task_name not in task_registry:
        available = ", ".join(sorted(task_registry))
        raise ValueError(f"Unknown task '{task_name}'. Available tasks: {available}")
    task_module = task_registry[task_name]
    task_object = task_module()
    # Run the evaluation
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: base|sft|mid|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # multiple choice 1 of 4 => 25%
        'ARC-Challenge': 0.25, # multiple choice 1 of 4 => 25%
        'MMLU': 0.25, # multiple choice 1 of 4 => 25%
        'GSM8K': 0.0, # open-ended => 0%
        'HumanEval': 0.0, # open-ended => 0%
        'SpellingBee': 0.0, # open-ended => 0%
    }
    task_names = all_tasks if args.task_name is None else [
        name.strip() for name in args.task_name.split('|') if name.strip()
    ]

    # Run all the task evaluations sequentially
    results = {}
    for task_name in task_names:
        with autocast_ctx:
            acc = run_chat_eval(
                task_name,
                model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
            results[task_name] = acc
            print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # Log to report
    from nanochat.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    get_report().log(section="Chat evaluation " + args.source, data=[
        vars(args), # CLI args
        results,
        chatcore_metric_dict,
    ])

    compute_cleanup()
