import argparse
import subprocess
import tempfile
from pathlib import Path

from utils.config import ensure_directories, load_config
from utils.file_reader import read_records, write_jsonl
from utils.logger import setup_logger
from utils.progress import progress_bar


SOURCE_KEYS = ["source_code", "source", "func", "code"]
IR_KEYS = ["llvm_ir", "ir", "llvm"]
LABEL_KEYS = ["label", "target", "vulnerable"]
SPLIT_KEYS = ["split", "partition"]
LANGUAGE_KEYS = ["language", "lang"]


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


def can_generate_ir(language):
    return str(language).lower() in {"c", "cpp", "c++", "cc", "cxx"}


def clang_language(language):
    language = str(language).lower()
    if language in {"cpp", "c++", "cc", "cxx"}:
        return "c++", ".cpp"
    return "c", ".c"


def generate_llvm_ir(source_code, language):
    clang_lang, suffix = clang_language(language)
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / f"sample{suffix}"
        ir_path = Path(temp_dir) / "sample.ll"
        source_path.write_text(source_code, encoding="utf-8")
        command = [
            "clang",
            "-S",
            "-emit-llvm",
            "-O0",
            "-x",
            clang_lang,
            str(source_path),
            "-o",
            str(ir_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "clang failed")
        return ir_path.read_text(encoding="utf-8")


def clean_record(record, index, logger):
    source_code = str(first_present(record, SOURCE_KEYS)).strip()
    llvm_ir = str(first_present(record, IR_KEYS)).strip()
    language = str(first_present(record, LANGUAGE_KEYS, "c")).strip().lower()
    split = str(first_present(record, SPLIT_KEYS, "train")).strip().lower()
    sample_id = str(record.get("sample_id", record.get("idx", index)))

    if not source_code:
        return None

    try:
        label = normalize_label(first_present(record, LABEL_KEYS))
    except ValueError as error:
        logger.info("Skipping %s: %s", sample_id, error)
        return None

    if not llvm_ir:
        if not can_generate_ir(language):
            logger.info("Skipping %s: missing LLVM IR for language '%s'", sample_id, language)
            return None
        try:
            llvm_ir = generate_llvm_ir(source_code, language)
        except Exception as error:
            logger.info("Skipping %s: LLVM IR generation failed: %s", sample_id, error)
            return None

    return {
        "sample_id": sample_id,
        "source_code": source_code,
        "llvm_ir": llvm_ir,
        "label": label,
        "language": language,
        "split": split,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess source and LLVM-IR dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    logger = setup_logger("preprocess", config["paths"]["logs"])

    raw_path = config["paths"]["raw_data"]
    processed_path = config["paths"]["processed_data"]
    records = read_records(raw_path)

    cleaned = []
    for index, record in progress_bar(enumerate(records), desc="Preprocessing", total=len(records)):
        cleaned_record = clean_record(record, index, logger)
        if cleaned_record is not None:
            cleaned.append(cleaned_record)

    write_jsonl(cleaned, processed_path)
    logger.info("Saved %d cleaned records to %s", len(cleaned), processed_path)


if __name__ == "__main__":
    main()
