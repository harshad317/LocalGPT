"""
Evaluate the CORE metric for a given model.

Run on a single GPU:
python -m scripts.base_eval

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 -m scripts.base_eval

The script will print the CORE metric to the console.
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
from contextlib import nullcontext

import torch

try:
    from nanochat.common import (
        compute_init,
        compute_cleanup,
        print0,
        get_base_dir,
        autodetect_device_type,
        download_file_with_lock,
    )
except ImportError:
    import ssl
    import urllib.request
    from contextlib import nullcontext

    def print0(s="", **kwargs):
        ddp_rank = int(os.environ.get("RANK", 0))
        if ddp_rank == 0:
            print(s, **kwargs)

    def get_base_dir():
        if os.environ.get("NANOCHAT_BASE_DIR"):
            nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
        else:
            nanochat_dir = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
        os.makedirs(nanochat_dir, exist_ok=True)
        return nanochat_dir

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
        return False, 0, 0, 1, torch.device(device_type)

    def compute_cleanup():
        return

    def download_file_with_lock(url, filename, postprocess_fn=None):
        base_dir = get_base_dir()
        file_path = os.path.join(base_dir, filename)
        lock_path = file_path + ".lock"

        if os.path.exists(file_path):
            return file_path

        try:
            from filelock import FileLock
        except Exception:
            FileLock = None

        lock_ctx = FileLock(lock_path) if FileLock is not None else nullcontext()
        with lock_ctx:
            if os.path.exists(file_path):
                return file_path

            ssl_context = None
            try:
                import certifi
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ssl_context = None

            try:
                with urllib.request.urlopen(url, context=ssl_context) as response, open(file_path, "wb") as f:
                    shutil.copyfileobj(response, f)
            except Exception:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                if shutil.which("curl"):
                    subprocess.run(
                        ["curl", "-L", "--fail", "--retry", "3", "-o", file_path, url],
                        check=True,
                    )
                else:
                    raise

            if postprocess_fn is not None:
                postprocess_fn(file_path)
            return file_path
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task

# -----------------------------------------------------------------------------
# nanochat specific function dealing with I/O etc.

# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle to disk (and unzip if needed)
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values from eval metadata
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the ability to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM
    if device.type == "cuda":
        torch_dtype = torch.bfloat16
    elif device.type == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    # transformers has changed the kwarg name from torch_dtype -> dtype in some versions
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        except TypeError:
            # Older transformers versions may not support some kwargs.
            model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # Load the tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path to evaluate')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task to evaluate (-1 = disable)')
    parser.add_argument('--model-tag', type=str, default=None, help='optional model tag for the output directory name')
    parser.add_argument('--step', type=str, default=None, help='optional model step for the output directory name')
    parser.add_argument('--device-type', type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help='device type to use (default: auto)')
    parser.add_argument('--fallback-hf-path', type=str, default="openai-community/gpt2", help='HF model path to use if no local checkpoints are found')
    parser.add_argument('--no-hf-fallback', action='store_true', help='disable HF fallback when local checkpoints are missing')
    args = parser.parse_args()

    # distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == "auto" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model and tokenizer from command line or from file system
    if args.hf_path is not None:
        # atm assume that if a path is given, it's a huggingface model path
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path # just for logging
        model_slug = hf_path.replace("/", "-") # for the output csv file
    else:
        # load a local model from the file system
        try:
            model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
            model_name = f"base_model (step {meta['step']})" # just for logging
            model_slug = f"base_model_{meta['step']:06d}" # for the output csv file
        except FileNotFoundError:
            base_dir = get_base_dir()
            checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
            if args.no_hf_fallback:
                raise SystemExit(
                    "No local base checkpoints found.\n"
                    f"- Expected: {checkpoints_dir}\n"
                    "- Fix: train a base model via `python -m scripts.base_train`, or run eval on a HF model via `python -m scripts.base_eval --hf-path openai-community/gpt2`."
                )
            if not args.fallback_hf_path:
                raise SystemExit(
                    "No local base checkpoints found and `--fallback-hf-path` is empty.\n"
                    f"- Expected: {checkpoints_dir}\n"
                    "- Fix: train a base model via `python -m scripts.base_train`, or pass `--hf-path <model>`."
                )
            print0(
                "No local base checkpoints found.\n"
                f"- Expected: {checkpoints_dir}\n"
                f"- Falling back to HuggingFace model: {args.fallback_hf_path}\n"
                "- Tip: pass `--no-hf-fallback` to disable this behavior."
            )
            hf_path = args.fallback_hf_path
            model, tokenizer = load_hf_model(hf_path, device)
            model_name = hf_path
            model_slug = hf_path.replace("/", "-")

    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # Print the content of the csv file to console too
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

    # Log to report
    from nanochat.report import get_report
    get_report().log(section="Base model evaluation", data=[
        {
            "Model": model_name,
            "CORE metric": core_metric,
        },
        centered_results, # the full table
    ])

    compute_cleanup()

if __name__ == "__main__":
    main()
