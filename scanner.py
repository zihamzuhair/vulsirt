import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models import build_model
from train import model_inputs, move_batch_to_device
from utils.config import ensure_directories, load_config
from utils.llvm import generate_llvm_ir


def detect_language(source_file):
    suffix = Path(source_file).suffix.lower()
    if suffix in {".cpp", ".cc", ".cxx"}:
        return "cpp"
    return "c"


def tokenize_single(tokenizer, source_code, llvm_ir, config):
    source_tokens = tokenizer(
        source_code,
        max_length=config["model"]["source_max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ir_tokens = tokenizer(
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
    if baseline in {"b2", "b3", "b4"}:
        llvm_ir = generate_llvm_ir(source_code, detect_language(args.source_file))

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    batch = tokenize_single(tokenizer, source_code, llvm_ir, config)
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
    print(f"Prediction: {'Vulnerable' if prediction == 1 else 'Non-vulnerable'}")
    print(f"Probability: {probability:.2f}")

    if baseline == "b4" and "alpha" in output:
        source_weight = output["alpha"].mean().item()
        ir_weight = 1 - source_weight
        print(f"Source weight: {source_weight:.2f}")
        print(f"LLVM-IR weight: {ir_weight:.2f}")


if __name__ == "__main__":
    main()
