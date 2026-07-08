"""PyTorch dataset wrapper for processed source/LLVM vulnerability records."""

import torch
from torch.utils.data import Dataset

from helpers.file_reader import read_jsonl


def split_matches(record_split, requested_split):
    """Compare split names while accepting validation aliases like val/valid."""
    record_split = str(record_split).lower()
    requested_split = str(requested_split).lower()
    if requested_split in {"validation", "valid", "val"}:
        return record_split in {"validation", "valid", "val"}
    if requested_split == "test":
        return "test" in record_split
    return record_split == requested_split


class VulnerabilityDataset(Dataset):
    """Load processed JSONL rows and tokenize source plus LLVM IR on demand."""

    def __init__(self, data_path, split, source_tokenizer, source_max_length, ir_max_length, config=None, ir_tokenizer=None):
        """Keep only the requested split and store tokenizer settings."""
        records = read_jsonl(data_path)
        self.records = [record for record in records if split_matches(record.get("split", ""), split)]
        self.source_tokenizer = source_tokenizer
        self.ir_tokenizer = ir_tokenizer or source_tokenizer
        self.source_max_length = source_max_length
        self.ir_max_length = ir_max_length

    def __len__(self):
        """Return how many records are available for this split."""
        return len(self.records)

    def _tokenize(self, tokenizer, text, max_length):
        """Tokenize text to a fixed length tensor for the model."""
        return tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, index):
        """Return one tokenized record and its label."""
        record = self.records[index]
        source_tokens = self._tokenize(self.source_tokenizer, record["source_code"], self.source_max_length)
        ir_tokens = self._tokenize(self.ir_tokenizer, record["llvm_ir"], self.ir_max_length)

        return {
            "sample_id": record["sample_id"],
            "source_input_ids": source_tokens["input_ids"].squeeze(0),
            "source_attention_mask": source_tokens["attention_mask"].squeeze(0),
            "ir_input_ids": ir_tokens["input_ids"].squeeze(0),
            "ir_attention_mask": ir_tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(float(record["label"]), dtype=torch.float),
        }
