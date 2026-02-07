"""
Quick MMLU subject distribution report.

Usage:
  python -m scripts.mmlu_subject_stats --subset=auxiliary_train --split=train
"""

import argparse
from collections import Counter

from tasks.mmlu import MMLU, MMLU_SUBJECT_GROUPS, MMLU_SUBJECTS


def _print_table(rows, total):
    for name, count in rows:
        pct = 100 * count / total if total else 0.0
        print(f"{name:30s} {count:8d}  {pct:6.2f}%")


def main():
    parser = argparse.ArgumentParser(description="MMLU subject distribution")
    parser.add_argument("--subset", type=str, default="auxiliary_train", choices=["all", "auxiliary_train"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--top", type=int, default=0, help="show only top N subjects (0 = all)")
    parser.add_argument("--upweight-physics", type=int, default=1)
    parser.add_argument("--upweight-biology", type=int, default=1)
    parser.add_argument("--upweight-engineering", type=int, default=1)
    parser.add_argument("--upweight-cs", type=int, default=1)
    parser.add_argument("--upweight-it", type=int, default=1)
    args = parser.parse_args()

    dataset = MMLU(subset=args.subset, split=args.split)

    def _count_subjects(ds):
        if "subject" in ds.ds.column_names:
            return Counter(row.get("subject") for row in ds.ds)
        return Counter(ds[i].get("subject") for i in range(len(ds)))

    counts = _count_subjects(dataset)
    total = sum(counts.values())

    print(f"Total rows: {total}")
    rows = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if args.top > 0:
        rows = rows[: args.top]
    _print_table(rows, total)

    missing_subjects = sum(v for k, v in counts.items() if not k)
    if missing_subjects > 0:
        print("\nWarning: missing subject for some rows. Columns:", dataset.ds.column_names)

    # If auxiliary_train has no subject labels, fall back to another split for stats.
    has_any_subject = any(k in MMLU_SUBJECTS for k in counts.keys())
    if args.subset == "auxiliary_train" and not has_any_subject:
        print("\nSubject labels missing in auxiliary_train; trying subset=all split=dev/validation/test for stats.")
        fallback = None
        for split_name in ("dev", "validation", "test", "auxiliary_train"):
            try:
                candidate = MMLU(subset="all", split=split_name)
            except Exception:
                continue
            cand_counts = _count_subjects(candidate)
            if any(k in MMLU_SUBJECTS for k in cand_counts.keys()):
                fallback = (candidate, cand_counts, split_name)
                break
        if fallback is None:
            print("Could not find subject labels in fallback splits.")
        else:
            dataset, counts, split_name = fallback
            total = sum(counts.values())
            rows = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
            if args.top > 0:
                rows = rows[: args.top]
            print(f"\nTotal rows (fallback split={split_name}): {total}")
            _print_table(rows, total)

    print("\nGrouped totals (base):")
    group_counts = {}
    for name, subjects in MMLU_SUBJECT_GROUPS.items():
        group_counts[name] = sum(counts.get(s, 0) for s in subjects)
    _print_table(sorted(group_counts.items()), total)

    weights = {
        "physics": args.upweight_physics,
        "biology": args.upweight_biology,
        "engineering": args.upweight_engineering,
        "cs": args.upweight_cs,
        "it": args.upweight_it,
    }
    if any(v > 1 for v in weights.values()):
        effective_total = total
        effective_group_counts = dict(group_counts)
        for name, weight in weights.items():
            if weight <= 1:
                continue
            extra = (weight - 1) * group_counts.get(name, 0)
            effective_group_counts[name] = effective_group_counts.get(name, 0) + extra
            effective_total += extra
        print("\nGrouped totals (with upweights):")
        _print_table(sorted(effective_group_counts.items()), effective_total)
        print(f"Effective total rows: {effective_total}")


if __name__ == "__main__":
    main()
