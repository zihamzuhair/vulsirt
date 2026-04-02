import torch
from torch.utils.data import Dataset

from utils.file_reader import read_jsonl


class VulnerabilityDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, source_max_length, ir_max_length):
        self.records = [
            record
            for record in read_jsonl(data_path)
            if self._matches_split(record.get("split", ""), split)
        ]
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
