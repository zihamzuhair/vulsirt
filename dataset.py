import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from utils.file_reader import read_jsonl


class VulnerabilityDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, source_max_length, ir_max_length, config=None):
        records = read_jsonl(data_path)
        self.records = self._records_for_split(records, split, config or {})
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.ir_max_length = ir_max_length

    @staticmethod
    def _matches_split(record_split, requested_split):
        record_split = str(record_split).lower()
        requested_split = str(requested_split).lower()
        if requested_split in {"validation", "valid", "val"}:
            return record_split in {"validation", "valid", "val"}
        if requested_split == "test":
            return "test" in record_split
        return record_split == requested_split

    @classmethod
    def _records_for_split(cls, records, split, config):
        data_config = config.get("data", {})
        records = cls._apply_sampling(records, data_config, config)

        split_config = data_config.get("split", {})
        split_mode = str(split_config.get("mode", "record")).lower()
        if split_mode in {"config", "ratio", "ratios", "generated"}:
            buckets = cls._split_by_config(records, split_config, data_config, config)
            requested_split = cls._canonical_split(split)
            return buckets.get(requested_split, [])

        return [
            record
            for record in records
            if cls._matches_split(record.get("split", ""), split)
        ]

    @staticmethod
    def _seed(data_config, config):
        return int(data_config.get("seed", config.get("training", {}).get("seed", 42)))

    @classmethod
    def _apply_sampling(cls, records, data_config, config):
        records = list(records)
        seed = cls._seed(data_config, config)
        rng = random.Random(seed)
        balance_config = data_config.get("balance", {})
        max_records = data_config.get("max_records")

        if balance_config.get("enabled", False):
            records = cls._balance_records(records, balance_config, max_records, rng)
        elif max_records:
            rng.shuffle(records)
            records = records[: int(max_records)]

        return records

    @staticmethod
    def _label(record):
        return int(record.get("label", 0))

    @classmethod
    def _balance_records(cls, records, balance_config, max_records, rng):
        by_label = defaultdict(list)
        for record in records:
            by_label[cls._label(record)].append(record)

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
        selected.extend(cls._sample_label(by_label[0], non_vulnerable_target, strategy, rng))
        selected.extend(cls._sample_label(by_label[1], vulnerable_target, strategy, rng))
        rng.shuffle(selected)

        if max_records and len(selected) > int(max_records):
            selected = selected[: int(max_records)]
        return selected

    @staticmethod
    def _sample_label(records, target, strategy, rng):
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

    @staticmethod
    def _canonical_split(split):
        split = str(split).lower()
        if split in {"validation", "valid", "val"}:
            return "validation"
        if split == "test":
            return "test"
        return "train"

    @classmethod
    def _split_by_config(cls, records, split_config, data_config, config):
        ratios = split_config.get("ratios", {})
        ratios = {
            "train": float(ratios.get("train", 0.8)),
            "validation": float(ratios.get("validation", ratios.get("val", 0.1))),
            "test": float(ratios.get("test", 0.1)),
        }
        total_ratio = sum(ratios.values())
        if total_ratio <= 0:
            raise ValueError("Configured split ratios must sum to a positive value.")
        ratios = {key: value / total_ratio for key, value in ratios.items()}

        seed = cls._seed(data_config, config)
        rng = random.Random(seed)
        buckets = {"train": [], "validation": [], "test": []}

        if split_config.get("stratify_by_label", True):
            grouped = defaultdict(list)
            for record in records:
                grouped[cls._label(record)].append(record)
            groups = grouped.values()
        else:
            groups = [list(records)]

        for group in groups:
            group = list(group)
            rng.shuffle(group)
            counts = cls._split_counts(len(group), ratios)
            train_count = counts["train"]
            validation_count = counts["validation"]
            buckets["train"].extend(group[:train_count])
            buckets["validation"].extend(group[train_count: train_count + validation_count])
            buckets["test"].extend(group[train_count + validation_count:])

        for bucket in buckets.values():
            rng.shuffle(bucket)
        return buckets

    @staticmethod
    def _split_counts(total, ratios):
        names = ["train", "validation", "test"]
        exact_counts = {name: total * ratios[name] for name in names}
        counts = {name: int(exact_counts[name]) for name in names}
        remaining = total - sum(counts.values())
        remainders = sorted(
            names,
            key=lambda name: exact_counts[name] - counts[name],
            reverse=True,
        )
        for index in range(remaining):
            counts[remainders[index % len(remainders)]] += 1
        return counts

    def __len__(self):
        return len(self.records)

    def _tokenize(self, text, max_length):
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, index):
        record = self.records[index]
        source_tokens = self._tokenize(record["source_code"], self.source_max_length)
        ir_tokens = self._tokenize(record["llvm_ir"], self.ir_max_length)

        return {
            "sample_id": record["sample_id"],
            "source_input_ids": source_tokens["input_ids"].squeeze(0),
            "source_attention_mask": source_tokens["attention_mask"].squeeze(0),
            "ir_input_ids": ir_tokens["input_ids"].squeeze(0),
            "ir_attention_mask": ir_tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(float(record["label"]), dtype=torch.float),
        }
