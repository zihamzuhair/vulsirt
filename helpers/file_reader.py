"""Small file helpers for reading and writing dataset records."""

import json
from pathlib import Path

import pandas as pd


def read_json(path):
    """Read a JSON file and always return a list of records."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    return [data]


def read_jsonl(path):
    """Read non-empty lines from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv(path):
    """Read a CSV file into a list of dictionaries."""
    return pd.read_csv(path).to_dict(orient="records")


def read_records(path):
    """Read JSON, JSONL, CSV, or a directory of JSONL files."""
    path = Path(path)
    if path.is_dir():
        records = []
        for child_path in sorted(path.glob("*.jsonl")):
            child_records = read_jsonl(child_path)
            split = split_from_filename(child_path.name)
            if split:
                for record in child_records:
                    record.setdefault("split", split)
            records.extend(child_records)
        return records

    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        return read_json(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def split_from_filename(filename):
    """Guess train/validation/test split from a file name."""
    filename = filename.lower()
    if "train" in filename:
        return "train"
    if "valid" in filename or "val" in filename:
        return "validation"
    if "test" in filename:
        return "test"
    return None


def write_jsonl(records, path):
    """Write records as JSONL, creating the parent folder if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
