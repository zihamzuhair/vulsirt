import random
from collections import defaultdict


LABEL_KEYS = ["label", "target", "vulnerable"]


def first_present(record, keys, default=""):
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def normalize_label(value):
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


def label_from_record(record):
    return normalize_label(first_present(record, LABEL_KEYS))


def seed_from_config(config):
    data_config = config.get("data", {})
    return int(data_config.get("seed", config.get("training", {}).get("seed", 42)))


def dataset_selection_config(config, dataset_name=None):
    data_config = dict(config.get("data", {}))
    dataset_overrides = data_config.get("datasets", {})
    if dataset_name and dataset_name in dataset_overrides:
        override = dataset_overrides[dataset_name] or {}
        merged = dict(data_config)
        merged.update({key: value for key, value in override.items() if key != "balance"})
        merged["balance"] = dict(data_config.get("balance", {}))
        merged["balance"].update(override.get("balance", {}) or {})
        return merged
    return data_config


def select_records(records, config, dataset_name=None):
    selection_config = dataset_selection_config(config, dataset_name)
    max_records = selection_config.get("max_records")
    balance_config = selection_config.get("balance", {})
    if not max_records and not balance_config.get("enabled", False):
        return list(records)

    rng = random.Random(seed_from_config(config))
    records = list(records)

    if not balance_config.get("enabled", False):
        rng.shuffle(records)
        return records[: int(max_records)]

    by_label = defaultdict(list)
    for record in records:
        try:
            by_label[label_from_record(record)].append(record)
        except ValueError:
            continue

    vulnerable_target = balance_config.get("vulnerable")
    non_vulnerable_target = balance_config.get("non_vulnerable")
    if vulnerable_target is None and non_vulnerable_target is None:
        if max_records:
            non_vulnerable_target = int(max_records) // 2
            vulnerable_target = int(max_records) - non_vulnerable_target
        else:
            shared_count = min(len(by_label[0]), len(by_label[1]))
            non_vulnerable_target = shared_count
            vulnerable_target = shared_count

    strategy = str(balance_config.get("strategy", "undersample")).lower()
    selected = []
    selected.extend(sample_label(by_label[0], non_vulnerable_target, strategy, rng))
    selected.extend(sample_label(by_label[1], vulnerable_target, strategy, rng))
    rng.shuffle(selected)
    if max_records and len(selected) > int(max_records):
        selected = selected[: int(max_records)]
    return selected


def sample_label(records, target, strategy, rng):
    records = list(records)
    rng.shuffle(records)
    if target is None:
        return records

    target = int(target)
    if target <= len(records):
        return records[:target]
    if strategy == "oversample" and records:
        return records + [rng.choice(records) for _ in range(target - len(records))]
    return records


def should_generate_splits(config):
    split_mode = str(config.get("data", {}).get("split", {}).get("mode", "record")).lower()
    return split_mode in {"config", "ratio", "ratios", "generated"}


def apply_config_splits(records, config):
    if not should_generate_splits(config):
        return list(records)

    data_config = config.get("data", {})
    split_config = data_config.get("split", {})
    ratios = normalized_ratios(split_config)
    rng = random.Random(seed_from_config(config))
    split_records = []

    if split_config.get("stratify_by_label", True):
        grouped = defaultdict(list)
        for record in records:
            grouped[int(record["label"])].append(record)
        groups = grouped.values()
    else:
        groups = [list(records)]

    for group in groups:
        group = list(group)
        rng.shuffle(group)
        counts = split_counts(len(group), ratios)
        train_end = counts["train"]
        validation_end = counts["train"] + counts["validation"]
        for index, record in enumerate(group):
            record = dict(record)
            if index < train_end:
                record["split"] = "train"
            elif index < validation_end:
                record["split"] = "validation"
            else:
                record["split"] = "test"
            split_records.append(record)

    rng.shuffle(split_records)
    return split_records


def normalized_ratios(split_config):
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


def split_counts(total, ratios):
    names = ["train", "validation", "test"]
    exact_counts = {name: total * ratios[name] for name in names}
    counts = {name: int(exact_counts[name]) for name in names}
    remaining = total - sum(counts.values())
    remainders = sorted(names, key=lambda name: exact_counts[name] - counts[name], reverse=True)
    for index in range(remaining):
        counts[remainders[index % len(remainders)]] += 1
    return counts


def split_matches(record_split, requested_split):
    record_split = str(record_split).lower()
    requested_split = str(requested_split).lower()
    if requested_split in {"validation", "valid", "val"}:
        return record_split in {"validation", "valid", "val"}
    if requested_split == "test":
        return "test" in record_split
    return record_split == requested_split
