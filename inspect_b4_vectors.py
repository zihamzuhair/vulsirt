import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models import build_model, first_token_features
from preprocess import generate_llvm_ir
from scanner import detect_language
from utils.config import load_config
from utils.file_reader import read_jsonl


def preview_tensor(name, tensor, values=8):
    flat = tensor.detach().cpu().flatten()
    preview = flat[:values].tolist()
    rounded = [round(value, 6) for value in preview]
    print(f"{name}")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  first {len(rounded)} values: {rounded}")


def tokenize_pair(tokenizer, source_code, llvm_ir, config, device):
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
        "source_input_ids": source_tokens["input_ids"].to(device),
        "source_attention_mask": source_tokens["attention_mask"].to(device),
        "ir_input_ids": ir_tokens["input_ids"].to(device),
        "ir_attention_mask": ir_tokens["attention_mask"].to(device),
    }


def load_sample_from_dataset(config, sample_index):
    records = read_jsonl(config["paths"]["processed_data"])
    if not records:
        raise ValueError(f"No records found in {config['paths']['processed_data']}")
    if sample_index < 0 or sample_index >= len(records):
        raise IndexError(f"sample_index must be between 0 and {len(records) - 1}")
    record = records[sample_index]
    return record["source_code"], record["llvm_ir"], record.get("sample_id", str(sample_index))


def load_sample_from_file(source_file):
    source_code = Path(source_file).read_text(encoding="utf-8")
    llvm_ir = generate_llvm_ir(source_code, detect_language(source_file))
    return source_code, llvm_ir, Path(source_file).name


def main():
    parser = argparse.ArgumentParser(description="Inspect B4 encoder, projection, gate, and classifier outputs.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--values", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = build_model("b4", config).to(device)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_checkpoint = Path(config["paths"]["checkpoints"]) / "b4_best.pt"
        checkpoint_path = str(default_checkpoint) if default_checkpoint.exists() else None

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint loaded. Showing outputs from an untrained B4 model.")

    if args.source_file:
        source_code, llvm_ir, sample_id = load_sample_from_file(args.source_file)
    else:
        source_code, llvm_ir, sample_id = load_sample_from_dataset(config, args.sample_index)

    print(f"Sample: {sample_id}")
    print(f"Model: {config['model']['name']}")
    print()

    batch = tokenize_pair(tokenizer, source_code, llvm_ir, config, device)
    model.eval()

    with torch.no_grad():
        source_output = model.source_encoder(
            input_ids=batch["source_input_ids"],
            attention_mask=batch["source_attention_mask"],
        )
        ir_output = model.ir_encoder(
            input_ids=batch["ir_input_ids"],
            attention_mask=batch["ir_attention_mask"],
        )

        source_features = first_token_features(source_output)
        ir_features = first_token_features(ir_output)

        source_projected = model.source_projection(source_features)
        ir_projected = model.ir_projection(ir_features)

        gate_input = torch.cat([source_projected, ir_projected], dim=1)
        alpha = torch.sigmoid(model.gate(gate_input))
        fused = alpha * source_projected + (1 - alpha) * ir_projected
        logits = model.classifier(model.dropout(fused)).squeeze(-1)
        probability = torch.sigmoid(logits)

    preview_tensor("source_features = source_encoder(source_code)", source_features, args.values)
    preview_tensor("ir_features = ir_encoder(llvm_ir)", ir_features, args.values)
    preview_tensor("source_projected = source_projection(source_features)", source_projected, args.values)
    preview_tensor("ir_projected = ir_projection(ir_features)", ir_projected, args.values)
    preview_tensor("alpha = sigmoid(gate(concat(source_projected, ir_projected)))", alpha, args.values)
    preview_tensor("fused = alpha * source_projected + (1 - alpha) * ir_projected", fused, args.values)
    preview_tensor("logit = classifier(fused)", logits, args.values)
    preview_tensor("probability = sigmoid(logit)", probability, args.values)

    source_weight = alpha.mean().item()
    ir_weight = 1 - source_weight
    print()
    print(f"Average source weight: {source_weight:.6f}")
    print(f"Average LLVM-IR weight: {ir_weight:.6f}")


if __name__ == "__main__":
    main()
