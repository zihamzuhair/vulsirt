import json
from pathlib import Path

import pandas as pd


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    return [data]


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_csv(path):
    return pd.read_csv(path).to_dict(orient="records")


def read_records(path):
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
    filename = filename.lower()
    if "train" in filename:
        return "train"
    if "valid" in filename or "val" in filename:
        return "validation"
    if "test" in filename:
        return "test"
    return None


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
