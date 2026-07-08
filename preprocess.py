import hashlib
import re
from collections import Counter
from pathlib import Path

from helpers.file_reader import read_records, write_jsonl
from helpers.logger import setup_logger


LABEL_KEYS = ["label", "target", "vulnerable"]
PROCESS_PRIMEVUL = True

PRIMEVUL_RAW_DIR = Path("data/raw/primevul")
PRIMEVUL_OUTPUT_JSONL = Path("data/processed/primevul_dataset.jsonl")

LOG_DIR = Path("logs")

PRIMEVUL_SPLIT_FILES = {
    "train": "primevul_train.jsonl",
    "validation": "primevul_valid.jsonl",
    "test": "primevul_test.jsonl",
}

EMPTY_FUNCTION_VALUES = {"", "none", "null", "nan", "na", "n/a", "undefined"}
MIN_FUNCTION_CHARS = 12
PRIMEVUL_KEEP_FILE_EXTENSIONS = {".c", ".h"}
PRIMEVUL_DISCARD_FILE_EXTENSIONS = {".cpp", ".cc", ".cxx", ".hpp", ".hh"}


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


def clean_function(value):
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def has_valid_function(function):
    lowered = function.lower()
    if lowered in EMPTY_FUNCTION_VALUES:
        return False
    if len(function) < MIN_FUNCTION_CHARS:
        return False
    return bool(re.search(r"[A-Za-z0-9_]", function))


def function_key(function):
    normalized = re.sub(r"\s+", " ", function).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def primevul_file_extension(raw_record):
    file_name = str(raw_record.get("file_name") or "").strip()
    if not file_name or file_name.lower() in EMPTY_FUNCTION_VALUES:
        return ""
    return Path(file_name).suffix.lower()


def base_processed_record(sample_id, source_code, label, language, split):
    return {
        "sample_id": str(sample_id),
        "source_code": source_code,
        "wrapped_source_code": "",
        "llvm_ir": "",
        "label": label,
        "language": language,
        "split": split,
    }


def clean_valid_function(raw_record, source_key):
    function = clean_function(raw_record.get(source_key))
    if not has_valid_function(function):
        return None, "missing_function"
    return function, None


def record_label(raw_record):
    try:
        return label_from_record(raw_record), None
    except ValueError:
        return None, "missing_label"


def add_mapped_record(record, skip_reason, records, seen_functions, stats, kept_key):
    if skip_reason:
        stats[skip_reason] += 1
        return

    key = function_key(record["source_code"])
    if key in seen_functions:
        stats["duplicate"] += 1
        return

    seen_functions.add(key)
    records.append(record)
    stats[kept_key] += 1


def map_source_record(raw_record, source_key, sample_id, language, split):
    function, skip_reason = clean_valid_function(raw_record, source_key)
    if skip_reason:
        return None, skip_reason

    label, skip_reason = record_label(raw_record)
    if skip_reason:
        return None, skip_reason

    return base_processed_record(sample_id, function, label, language, split), None


def map_primevul_record(raw_record, split, row_number):
    extension = primevul_file_extension(raw_record)
    if extension not in PRIMEVUL_KEEP_FILE_EXTENSIONS:
        if extension in PRIMEVUL_DISCARD_FILE_EXTENSIONS:
            return None, f"discarded_{extension[1:]}"
        return None, "discarded_non_c_file"

    return map_source_record(
        raw_record,
        "func",
        raw_record.get("idx", f"{split}_{row_number}"),
        "c",
        split,
    )


def preprocess_primevul(raw_dir, logger):
    raw_dir = Path(raw_dir)
    records = []
    seen_functions = set()
    stats = Counter()

    for split, filename in PRIMEVUL_SPLIT_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            logger.info("Missing PrimeVul %s file: %s", split, path)
            continue

        for row_number, raw_record in enumerate(read_records(path)):
            record, skip_reason = map_primevul_record(raw_record, split, row_number)
            add_mapped_record(record, skip_reason, records, seen_functions, stats, f"kept_{split}")

    logger.info(
        "PrimeVul preprocessing complete: kept=%d "
        "(train=%d, validation=%d, test=%d), removed_missing_function=%d, "
        "removed_missing_label=%d, removed_duplicates=%d, "
        "discarded_non_c_file=%d, discarded_cpp=%d, discarded_cc=%d, "
        "discarded_cxx=%d, discarded_hpp=%d, discarded_hh=%d",
        len(records),
        stats["kept_train"],
        stats["kept_validation"],
        stats["kept_test"],
        stats["missing_function"],
        stats["missing_label"],
        stats["duplicate"],
        stats["discarded_non_c_file"],
        stats["discarded_cpp"],
        stats["discarded_cc"],
        stats["discarded_cxx"],
        stats["discarded_hpp"],
        stats["discarded_hh"],
    )
    return records

def save_records(records, output_path, logger, dataset_name):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_path)
    print(f"Saved {len(records)} {dataset_name} records to {output_path}", flush=True)
    logger.info("Saved %d %s records to %s", len(records), dataset_name, output_path)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("preprocess", LOG_DIR)

    if PROCESS_PRIMEVUL:
        print("1. Preprocessing PrimeVul dataset", flush=True)
        print(f"   Input: {PRIMEVUL_RAW_DIR}", flush=True)
        print(f"   Output: {PRIMEVUL_OUTPUT_JSONL}", flush=True)
        logger.info("Preprocessing PrimeVul from %s", PRIMEVUL_RAW_DIR)
        primevul_records = preprocess_primevul(PRIMEVUL_RAW_DIR, logger)
        save_records(primevul_records, PRIMEVUL_OUTPUT_JSONL, logger, "PrimeVul")

    if not PROCESS_PRIMEVUL:
        print("No datasets selected. Enable PROCESS_PRIMEVUL.", flush=True)
        logger.info("No datasets selected. Enable PROCESS_PRIMEVUL.")

    print("Dataset preprocessing complete", flush=True)


if __name__ == "__main__":
    main()
