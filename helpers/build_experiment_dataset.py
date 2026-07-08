import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from helpers.config_loader import load_config, primevul_processed_path
from helpers.file_reader import read_jsonl, write_jsonl


LABEL_KEYS = ["label", "target", "vulnerable"]
DEFAULT_INPUT_JSONL = Path("data/processed/primevul_dataset_with_llvm.jsonl")
DEFAULT_CONFIG_PATH = Path("configs/100_samples.yaml")


def first_present(record: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "vulnerable", "vul"}:
        return 1
    if text in {"0", "false", "no", "non-vulnerable", "safe", "clean"}:
        return 0
    raise ValueError(f"Invalid label: {value}")


def label_from_record(record: dict[str, Any]) -> int:
    return normalize_label(first_present(record, LABEL_KEYS))


def seed_from_config(config: dict[str, Any]) -> int:
    data_config = config.get("data", {})
    return int(data_config.get("seed", config.get("training", {}).get("seed", 42)))


def normalized_ratios(split_config: dict[str, Any]) -> dict[str, float]:
    ratios = split_config.get("ratios", {})
    ratios = {
        "train": float(ratios.get("train", 0.8)),
        "validation": float(ratios.get("validation", ratios.get("val", 0.1))),
        "test": float(ratios.get("test", 0.1)),
    }
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("Configured split ratios must sum to a positive value.")
    return {key: value / total_ratio for key, value in ratios.items()}


def split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    names = ["train", "validation", "test"]
    exact_counts = {name: total * ratios[name] for name in names}
    counts = {name: int(exact_counts[name]) for name in names}
    remaining = total - sum(counts.values())
    remainders = sorted(names, key=lambda name: exact_counts[name] - counts[name], reverse=True)
    for index in range(remaining):
        counts[remainders[index % len(remainders)]] += 1
    return counts


def allocate_label_targets(
    requested_targets: dict[int, int],
    available_counts: dict[int, int],
) -> dict[int, int]:
    targets = {
        label: min(requested_targets[label], available_counts.get(label, 0))
        for label in requested_targets
    }

    shortfall = sum(requested_targets.values()) - sum(targets.values())
    if shortfall <= 0:
        return targets

    for label in sorted(targets, key=lambda value: requested_targets[value] - targets[value]):
        spare = max(0, available_counts.get(label, 0) - targets[label])
        fill = min(spare, shortfall)
        targets[label] += fill
        shortfall -= fill
        if shortfall == 0:
            break

    return targets


def configured_total(config: dict[str, Any]) -> int:
    primevul_config = config.get("data", {}).get("datasets", {}).get("primevul", {})
    total = primevul_config.get("max_records", config.get("data", {}).get("max_records"))
    if total is None:
        raise ValueError("Set data.datasets.primevul.max_records or data.max_records in the config.")
    total = int(total)
    if total < 1:
        raise ValueError("Configured max_records must be positive.")
    return total


def configured_label_targets(config: dict[str, Any], total: int) -> dict[int, int]:
    primevul_config = config.get("data", {}).get("datasets", {}).get("primevul", {})
    balance_config = primevul_config.get("balance", {})
    if not balance_config.get("enabled", False):
        vulnerable_target = total // 2
        return {1: vulnerable_target, 0: total - vulnerable_target}

    vulnerable = balance_config.get("vulnerable")
    non_vulnerable = balance_config.get("non_vulnerable")
    if vulnerable is None and non_vulnerable is None:
        vulnerable = total // 2
        non_vulnerable = total - vulnerable
    elif vulnerable is None:
        non_vulnerable = int(non_vulnerable)
        vulnerable = total - non_vulnerable
    elif non_vulnerable is None:
        vulnerable = int(vulnerable)
        non_vulnerable = total - vulnerable
    else:
        vulnerable = int(vulnerable)
        non_vulnerable = int(non_vulnerable)

    if vulnerable < 0 or non_vulnerable < 0:
        raise ValueError("Configured PrimeVul class targets must be non-negative.")
    if vulnerable + non_vulnerable != total:
        raise ValueError(
            "PrimeVul balance targets must sum to data.datasets.primevul.max_records "
            f"({vulnerable} + {non_vulnerable} != {total})."
        )
    return {1: vulnerable, 0: non_vulnerable}


def trainable_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for record in read_jsonl(path):
        if not str(record.get("source_code") or "").strip():
            continue
        if not str(record.get("llvm_ir") or "").strip():
            continue
        try:
            label = label_from_record(record)
        except ValueError:
            continue
        output = dict(record)
        output["label"] = label
        records.append(output)
    return records


def take(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected = records[:count]
    del records[:len(selected)]
    return selected


def balanced_fill(
    vulnerable: list[dict[str, Any]],
    non_vulnerable: list[dict[str, Any]],
    targets: dict[int, int],
) -> list[dict[str, Any]]:
    vulnerable_target = targets[1]
    non_vulnerable_target = targets[0]

    selected_vulnerable = take(vulnerable, vulnerable_target)
    selected_non_vulnerable = take(non_vulnerable, non_vulnerable_target)

    shortfall = sum(targets.values()) - len(selected_vulnerable) - len(selected_non_vulnerable)
    if shortfall > 0:
        if len(selected_vulnerable) < vulnerable_target:
            selected_non_vulnerable.extend(take(non_vulnerable, shortfall))
        else:
            selected_vulnerable.extend(take(vulnerable, shortfall))

    return selected_vulnerable + selected_non_vulnerable


def build_experiment_records(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    total = configured_total(config)
    requested_label_targets = configured_label_targets(config, total)
    ratios = normalized_ratios(config.get("data", {}).get("split", {}))
    rng = random.Random(seed_from_config(config))

    vulnerable = [record for record in records if int(record["label"]) == 1]
    non_vulnerable = [record for record in records if int(record["label"]) == 0]
    label_targets = allocate_label_targets(
        requested_label_targets,
        {1: len(vulnerable), 0: len(non_vulnerable)},
    )
    label_split_counts = {
        label: split_counts(target, ratios)
        for label, target in label_targets.items()
    }

    rng.shuffle(vulnerable)
    rng.shuffle(non_vulnerable)

    selected = []
    for split in ("train", "validation", "test"):
        split_records = balanced_fill(
            vulnerable,
            non_vulnerable,
            {label: counts[split] for label, counts in label_split_counts.items()},
        )
        for record in split_records:
            output = dict(record)
            output["split"] = split
            selected.append(output)

    rng.shuffle(selected)
    return selected


def output_path(config: dict[str, Any], output_jsonl: Path | None = None) -> Path:
    if output_jsonl is not None:
        return output_jsonl
    return Path(primevul_processed_path(config))


def build_for_config(config_path: Path, input_jsonl: Path, output_jsonl: Path | None = None) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    config = load_config(config_path)
    source_records = trainable_records(input_jsonl)
    total = configured_total(config)
    requested_label_targets = configured_label_targets(config, total)
    available_by_label = Counter(int(record["label"]) for record in source_records)
    experiment_records = build_experiment_records(source_records, config)
    destination = output_path(config, output_jsonl)
    write_jsonl(experiment_records, destination)

    counts = Counter((record["split"], int(record["label"])) for record in experiment_records)
    written_by_label = Counter(int(record["label"]) for record in experiment_records)
    return {
        "input": str(input_jsonl),
        "config": str(config_path),
        "output": str(destination),
        "available": len(source_records),
        "available_by_label": {
            "vulnerable": available_by_label[1],
            "non_vulnerable": available_by_label[0],
        },
        "requested_by_label": {
            "vulnerable": requested_label_targets[1],
            "non_vulnerable": requested_label_targets[0],
        },
        "shortfall_by_label": {
            "vulnerable": max(0, requested_label_targets[1] - written_by_label[1]),
            "non_vulnerable": max(0, requested_label_targets[0] - written_by_label[0]),
        },
        "written": len(experiment_records),
        "splits": {
            split: {
                "vulnerable": counts[(split, 1)],
                "non_vulnerable": counts[(split, 0)],
                "total": counts[(split, 1)] + counts[(split, 0)],
            }
            for split in ("train", "validation", "test")
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build config-sized PrimeVul experiment datasets.")
    parser.add_argument(
        "--config",
        nargs="+",
        type=Path,
        default=[DEFAULT_CONFIG_PATH],
        help="One or more config files to build. Defaults to configs/100_samples.yaml.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help="Compiled PrimeVul JSONL with LLVM IR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override output JSONL path. Only valid with one --config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"compiled input not found: {args.input}")
    if args.output is not None and len(args.config) != 1:
        raise SystemExit("--output can only be used with a single --config.")

    reports = [
        build_for_config(config_path, args.input, args.output)
        for config_path in args.config
    ]
    print(json.dumps(reports[0] if len(reports) == 1 else reports, indent=2))


if __name__ == "__main__":
    main()
