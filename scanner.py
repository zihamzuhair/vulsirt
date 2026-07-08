import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models import build_model
from compiler import compile_c_function, is_successful_llvm_record
from train import model_inputs, move_batch_to_device
from helpers.config_loader import ensure_directories, load_config, model_ir_name, model_source_name


def detect_language(source_file, source_code):
    suffix = Path(source_file).suffix.lower()
    if suffix in {".c", ".h"}:
        return "c"
    if suffix in {".rs"}:
        return "rust"
    if suffix in {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}:
        return "cpp"

    lowered = source_code.lower()
    if "fn " in lowered and ("let " in lowered or "pub " in lowered or "impl " in lowered):
        return "rust"
    if "#include" in source_code or "->" in source_code or ";" in source_code:
        if any(token in source_code for token in ("int ", "void ", "char ", "struct ", "return ")):
            return "c"
    return "unsupported"


def compile_for_scan(source_file, source_code):
    language = detect_language(source_file, source_code)
    if language == "c":
        compiled = compile_c_function(
            source_code,
            file_name=Path(source_file).name,
            sample_id=Path(source_file).stem or "scanner_input",
        )
        if not is_successful_llvm_record(compiled):
            error = compiled.get("compile_error") or compiled.get("ir_status") or "LLVM generation failed"
            raise RuntimeError(f"C compilation failed: {error}")
        return compiled
    if language == "rust":
        raise NotImplementedError("Rust scanner compilation is not implemented yet.")
    if language == "cpp":
        raise ValueError("C++ input is not supported by the scanner compiler path.")
    raise ValueError("Unsupported input language. Use a C source file for now.")


def tokenize_single(source_tokenizer, ir_tokenizer, source_code, llvm_ir, config):
    source_tokens = source_tokenizer(
        source_code,
        max_length=config["model"]["source_max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ir_tokens = ir_tokenizer(
        llvm_ir,
        max_length=config["model"]["ir_max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "source_input_ids": source_tokens["input_ids"],
        "source_attention_mask": source_tokens["attention_mask"],
        "ir_input_ids": ir_tokens["input_ids"],
        "ir_attention_mask": ir_tokens["attention_mask"],
        "label": torch.tensor([0.0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Scan one source file with a trained model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline", choices=["b1", "b2", "b3", "b4"], required=True)
    parser.add_argument("--source-file", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    baseline = args.baseline.lower()
    checkpoint_path = Path(config["paths"]["checkpoints"]) / f"{baseline}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    source_code = Path(args.source_file).read_text(encoding="utf-8")
    llvm_ir = ""
    compiled = None
    if baseline in {"b2", "b3", "b4"}:
        compiled = compile_for_scan(args.source_file, source_code)
        source_code = compiled.get("source_code", source_code)
        llvm_ir = compiled["llvm_ir"]

    source_tokenizer = AutoTokenizer.from_pretrained(model_source_name(config))
    ir_tokenizer = AutoTokenizer.from_pretrained(model_ir_name(config))
    batch = tokenize_single(source_tokenizer, ir_tokenizer, source_code, llvm_ir, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(baseline, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        batch = move_batch_to_device(batch, device)
        output = model(**model_inputs(batch))
        probability = torch.sigmoid(output["logits"]).item()
        prediction = 1 if probability >= config["training"]["threshold"] else 0

    print(f"Baseline: {baseline.upper()}")
    if compiled:
        print(f"Detected language: {compiled.get('language', 'c')}")
        print(f"Compile status: {compiled.get('ir_status', 'success')}")
        if compiled.get("wrapped_source_code"):
            print("Wrapped source: generated")
    print(f"Prediction: {'Vulnerable' if prediction == 1 else 'Non-vulnerable'}")
    print(f"Probability: {probability:.2f}")

    if baseline == "b4" and "alpha" in output:
        source_weight = output["alpha"].mean().item()
        ir_weight = 1 - source_weight
        print(f"Source weight: {source_weight:.2f}")
        print(f"LLVM-IR weight: {ir_weight:.2f}")


if __name__ == "__main__":
    main()
