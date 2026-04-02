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
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        return read_json(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
