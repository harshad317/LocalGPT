"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the FineWeb-Edu dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

try:
    from nanochat.common import get_base_dir
except ImportError:
    def get_base_dir():
        # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
        if os.environ.get("NANOCHAT_BASE_DIR"):
            nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
        else:
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".cache")
            nanochat_dir = os.path.join(cache_dir, "nanochat")
        os.makedirs(nanochat_dir, exist_ok=True)
        return nanochat_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining datasets
base_dir = get_base_dir()

DATASET_CONFIGS = {
    "fineweb-edu": {
        "base_url": "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        "num_shards": 1823,  # shards 00000..01822
        "filename_template": "shard_{index:05d}.parquet",
        "data_dir": os.path.join(base_dir, "base_data"),  # keep legacy path
    },
    "finemath-4plus": {
        "base_url": "https://huggingface.co/datasets/HuggingFaceTB/finemath/resolve/main/finemath-4plus",
        "num_shards": 64,  # train-00000-of-00064..train-00063-of-00064
        "filename_template": "train-{index:05d}-of-00064.parquet",
        "data_dir": os.path.join(base_dir, "finemath-4plus"),
    },
    "med_data": {
        "base_url": "https://huggingface.co/datasets/harshad317/MED_data/resolve/main/data",
        "num_shards": 13,  # train-00000-of-00013..train-00012-of-00013
        "filename_template": "train-{index:05d}-of-00013.parquet",
        "data_dir": os.path.join(base_dir, "med_data"),
    },
}

DEFAULT_DATASETS = ("fineweb-edu", "finemath-4plus", "med_data")
DEFAULT_DOWNLOAD_DATASETS = ("fineweb-edu",)
DATASET_ALIASES = {
    "fineweb": "fineweb-edu",
    "fineweb-edu-100b": "fineweb-edu",
    "fineweb-edu-100b-shuffle": "fineweb-edu",
    "karpathy/fineweb-edu-100b-shuffle": "fineweb-edu",
    "finemath": "finemath-4plus",
    "HuggingFaceTB/finemath": "finemath-4plus",
    "med-data": "med_data",
    "harshad317/MED_data": "med_data",
}

for cfg in DATASET_CONFIGS.values():
    os.makedirs(cfg["data_dir"], exist_ok=True)


def _normalize_dataset_names(dataset_names=None, default=None):
    if dataset_names is None:
        env_datasets = os.environ.get("NANOCHAT_BASE_DATASETS")
        if env_datasets:
            dataset_names = env_datasets
        elif default is not None:
            dataset_names = default
        else:
            dataset_names = DEFAULT_DATASETS
    if isinstance(dataset_names, str):
        dataset_names = [name.strip() for name in dataset_names.split(",") if name.strip()]
    resolved = []
    seen = set()
    for name in dataset_names:
        canonical = DATASET_ALIASES.get(name, name)
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    unknown = [name for name in resolved if name not in DATASET_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Available: {sorted(DATASET_CONFIGS.keys())}")
    return resolved


def _index_to_filename(dataset_name, index):
    template = DATASET_CONFIGS[dataset_name]["filename_template"]
    return template.format(index=index)


def _list_parquet_files_in_dir(data_dir):
    if not os.path.isdir(data_dir):
        return []
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return [os.path.join(data_dir, f) for f in parquet_files]


def _split_train_val(paths):
    if len(paths) >= 2:
        return paths[:-1], [paths[-1]]
    return paths, []


def _interleave(paths_by_dataset):
    if not paths_by_dataset:
        return []
    max_len = max(len(paths) for paths in paths_by_dataset)
    interleaved = []
    for idx in range(max_len):
        for paths in paths_by_dataset:
            if idx < len(paths):
                interleaved.append(paths[idx])
    return interleaved

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, dataset_names=None, split=None):
    """Return parquet file paths for the requested split and dataset(s)."""
    assert split in [None, "train", "val"], "split must be None, 'train', or 'val'"
    if data_dir is not None:
        paths = _list_parquet_files_in_dir(data_dir)
        if split is None:
            return paths
        train_paths, val_paths = _split_train_val(paths)
        return train_paths if split == "train" else val_paths

    dataset_names = _normalize_dataset_names(dataset_names)
    per_dataset_paths = []
    val_paths = []
    for dataset_name in dataset_names:
        data_dir = DATASET_CONFIGS[dataset_name]["data_dir"]
        paths = _list_parquet_files_in_dir(data_dir)
        if not paths:
            continue
        train_paths, dataset_val_paths = _split_train_val(paths)
        if split is None:
            per_dataset_paths.append(paths)
        elif split == "train":
            per_dataset_paths.append(train_paths)
        elif split == "val":
            if dataset_val_paths:
                val_paths.extend(dataset_val_paths)

    if split == "val":
        return val_paths
    return _interleave(per_dataset_paths)

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last shard of each dataset is used for val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(split=split)
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(dataset_name, index):
    """Downloads a single dataset shard, with some backoff."""

    # Construct the local filepath for this file and skip if it already exists
    filename = _index_to_filename(dataset_name, index)
    data_dir = DATASET_CONFIGS[dataset_name]["data_dir"]
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    base_url = DATASET_CONFIGS[dataset_name]["base_url"]
    url = f"{base_url}/{filename}"
    print(f"Downloading {dataset_name}/{filename}...")

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    headers = {"Authorization": f"Bearer {token}"} if token else None

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {dataset_name}/{filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {dataset_name}/{filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {dataset_name}/{filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download base pretraining dataset shards")
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of shards to download per dataset (default: -1), -1 = all shards",
    )
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names (default: env NANOCHAT_BASE_DATASETS or fineweb-edu).",
    )
    args = parser.parse_args()

    dataset_names = _normalize_dataset_names(args.datasets, default=DEFAULT_DOWNLOAD_DATASETS)
    for dataset_name in dataset_names:
        cfg = DATASET_CONFIGS[dataset_name]
        total_shards = cfg["num_shards"]
        num = total_shards if args.num_files == -1 else min(args.num_files, total_shards)
        ids_to_download = list(range(num))
        print(f"Downloading {len(ids_to_download)} shards for {dataset_name} using {args.num_workers} workers...")
        print(f"Target directory: {cfg['data_dir']}")
        print()
        with Pool(processes=args.num_workers) as pool:
            results = pool.starmap(download_single_file, [(dataset_name, idx) for idx in ids_to_download])

        # Report results
        successful = sum(1 for success in results if success)
        print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {cfg['data_dir']}")
