import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dataset import apply_config_splits, label_from_record, select_records, should_generate_splits
from utils.config import (
    ensure_directories,
    load_config,
    primevul_processed_path,
    primevul_raw_dir,
    rust_processed_path,
    rust_raw_path,
)
from utils.file_reader import read_records, write_jsonl
from utils.logger import setup_logger
from utils.llvm import LLVMGenerationError, can_generate_ir, generate_llvm_ir, llvm_error_category
from utils.progress import progress_bar


SOURCE_KEYS = ["source_code", "source", "func", "code"]
IR_KEYS = ["llvm_ir", "ir", "llvm"]
SPLIT_KEYS = ["split", "partition"]
LANGUAGE_KEYS = ["language", "lang"]


def first_present(record, keys, default=""):
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def clean_record(record, index, logger, config=None, forced_split=None, forced_language=None, dataset_name="dataset"):
    config = config or {}
    source_code = str(first_present(record, SOURCE_KEYS)).strip()
    llvm_ir = str(first_present(record, IR_KEYS)).strip()
    language = str(forced_language or first_present(record, LANGUAGE_KEYS, "c")).strip().lower()
    split = str(forced_split or first_present(record, SPLIT_KEYS, "train")).strip().lower()
    sample_id = str(record.get("sample_id", record.get("idx", index)))
    ir_status = "provided" if llvm_ir else "missing"

    if not source_code:
        return None

    try:
        label = label_from_record(record)
    except ValueError as error:
        logger.info("Skipping %s: %s", sample_id, error)
        return None

    if not llvm_ir:
        should_generate_ir = config.get("preprocessing", {}).get("generate_missing_ir", True)
        if not should_generate_ir:
            llvm_ir = ir_fallback(source_code, config, language)
            if not llvm_ir:
                logger.info("Skipping %s: missing LLVM IR and fallback disabled", sample_id)
                return None
            ir_status = "fallback"
        elif not can_generate_ir(language):
            write_llvm_error(
                config,
                sample_id=sample_id,
                record_index=index,
                dataset_name=dataset_name,
                language=language,
                split=split,
                source_code=source_code,
                error=ValueError(f"Unsupported LLVM generation language: {language}"),
                stage="unsupported_language",
            )
            llvm_ir = ir_fallback(source_code, config, language)
            if not llvm_ir:
                logger.info("Skipping %s: missing LLVM IR for language '%s'", sample_id, language)
                return None
            ir_status = "fallback"
        try:
            if not llvm_ir:
                llvm_ir = generate_llvm_ir(source_code, language)
                ir_status = "generated"
        except Exception as error:
            write_llvm_error(
                config,
                sample_id=sample_id,
                record_index=index,
                dataset_name=dataset_name,
                language=language,
                split=split,
                source_code=source_code,
                error=error,
                stage="compiler_failure",
            )
            llvm_ir = ir_fallback(source_code, config, language)
            if not llvm_ir:
                logger.info("Skipping %s: LLVM IR generation failed: %s", sample_id, error)
                return None
            ir_status = "fallback"
            logger.info("Using LLVM IR fallback for %s sample %s: %s", dataset_name, sample_id, error)

    return {
        "sample_id": sample_id,
        "source_code": source_code,
        "llvm_ir": llvm_ir,
        "label": label,
        "language": language,
        "split": split,
        "dataset": dataset_name,
        "ir_status": ir_status,
    }


def ir_fallback(source_code, config, language):
    preprocessing = config.get("preprocessing", {})
    if str(language).lower() in {"rust", "rs"}:
        mode = str(preprocessing.get("rust_ir_failure", preprocessing.get("ir_failure", "skip"))).lower()
    else:
        mode = str(preprocessing.get("ir_failure", "skip")).lower()
    if mode == "source":
        return source_code
    if mode == "empty":
        return " "
    return ""


def llvm_error_path(config):
    return config.get("paths", {}).get("llvm_errors", str(Path(config["paths"]["logs"]) / "llvm_generation_errors.jsonl"))


def reset_llvm_error_file(config):
    path = Path(llvm_error_path(config))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()


def write_llvm_error(config, sample_id, record_index, dataset_name, language, split, source_code, error, stage):
    path = Path(llvm_error_path(config))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "record_index": record_index,
        "dataset": dataset_name,
        "language": language,
        "split": split,
        "stage": stage,
        "error_category": llvm_error_category(error, stage),
        "error_type": type(error).__name__,
        "error": str(error),
        "source_length": len(source_code),
        "source_preview": source_code[:500],
    }
    if isinstance(error, LLVMGenerationError):
        record.update(
            {
                "compiler": error.compiler,
                "returncode": error.returncode,
                "command": error.command,
                "stdout": error.stdout,
                "stderr": error.stderr,
            }
        )
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def preprocess_records(records, config, logger, dataset_name, forced_split=None, forced_language=None, apply_selection=True):
    cleaned = []
    records = select_records(records, config, dataset_name) if apply_selection else list(records)
    for index, record in progress_bar(
        enumerate(records),
        desc=f"Preprocessing {dataset_name}",
        total=len(records),
    ):
        cleaned_record = clean_record(
            record,
            index,
            logger,
            config,
            forced_split=forced_split,
            forced_language=forced_language,
            dataset_name=dataset_name,
        )
        if cleaned_record is not None:
            cleaned.append(cleaned_record)
    return cleaned


def path_exists(config, key):
    value = config.get("paths", {}).get(key)
    return bool(value) and Path(value).exists()


def preprocess_primevul(config, logger):
    raw_dir = Path(primevul_raw_dir(config))
    split_files = [
        ("train", raw_dir / "primevul_train.jsonl"),
        ("validation", raw_dir / "primevul_valid.jsonl"),
        ("test", raw_dir / "primevul_test.jsonl"),
    ]
    if should_generate_splits(config):
        combined_records = []
        for split, path in split_files:
            if not path.exists():
                logger.info("PrimeVul %s path not found: %s", split, path)
                continue
            records = read_records(path)
            for record in records:
                record = dict(record)
                record["original_split"] = split
                combined_records.append(record)

        selected_records = select_records(combined_records, config, "primevul")
        cleaned = preprocess_records(
            selected_records,
            config,
            logger,
            "primevul",
            forced_language="cpp",
            apply_selection=False,
        )
        return apply_config_splits(cleaned, config)

    all_records = []
    for split, path in split_files:
        if not path.exists():
            logger.info("PrimeVul %s path not found: %s", split, path)
            continue
        records = read_records(path)
        all_records.extend(
            preprocess_records(
                records,
                config,
                logger,
                "primevul",
                forced_split=split,
                forced_language="cpp",
            )
        )
    return all_records


def preprocess_rust(config, logger):
    raw_path = rust_raw_path(config)
    if not raw_path or not Path(raw_path).exists():
        logger.info("Rust path not found: %s", raw_path)
        return []
    records = read_records(raw_path)
    return preprocess_records(
        records,
        config,
        logger,
        "rust",
        forced_split="rust_test",
        forced_language="rust",
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess source and LLVM-IR dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["auto", "single", "primevul", "rust", "all"], default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    logger = setup_logger("preprocess", config["paths"]["logs"])
    reset_llvm_error_file(config)

    mode = args.dataset
    if mode == "auto":
        raw_primevul = config.get("paths", {}).get("raw_primevul")
        raw_rust = rust_raw_path(config)
        mode = "all" if (raw_primevul and Path(raw_primevul).exists()) or (raw_rust and Path(raw_rust).exists()) else "single"

    if mode in {"primevul", "all"}:
        primevul_records = preprocess_primevul(config, logger)
        primevul_path = primevul_processed_path(config)
        write_jsonl(primevul_records, primevul_path)
        logger.info("Saved %d PrimeVul records to %s", len(primevul_records), primevul_path)

    if mode in {"rust", "all"}:
        rust_records = preprocess_rust(config, logger)
        rust_path = rust_processed_path(config)
        write_jsonl(rust_records, rust_path)
        logger.info("Saved %d Rust records to %s", len(rust_records), rust_path)

    if mode == "single":
        raw_path = config["paths"]["raw_data"]
        processed_path = config["paths"]["processed_data"]
        records = preprocess_records(read_records(raw_path), config, logger, "dataset")
        records = apply_config_splits(records, config)
        write_jsonl(records, processed_path)
        logger.info("Saved %d cleaned records to %s", len(records), processed_path)


if __name__ == "__main__":
    main()
