import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import FEATURES_DIR, PROJECT_ROOT, ensure_ablation_dirs, resolve_dir

from dataset import VulnerabilityDataset
from helpers.config import load_config, model_ir_name, model_source_name, primevul_processed_path
from helpers.progress import progress_bar
from models import build_model, first_token_features


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def b4_fused_vectors(model, batch):
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
    if getattr(model, "gate_mode", "learned") == "fixed" and getattr(model, "fixed_alpha", None) is not None:
        fixed_alpha = min(max(float(model.fixed_alpha), 0.0), 1.0)
        alpha = torch.full_like(source_projected, fixed_alpha)
    else:
        alpha = torch.sigmoid(model.gate(gate_input) / model.gate_temperature)

    fused = alpha * source_projected + (1 - alpha) * ir_projected
    return fused, alpha


def load_b4_model(config, checkpoint_path, device):
    model = build_model("b4", config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_split(config, model, source_tokenizer, ir_tokenizer, split, output_dir, batch_size, device):
    data_path = Path(primevul_processed_path(config))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path or not Path(data_path).exists():
        raise FileNotFoundError(f"Processed PrimeVul data not found at {data_path}")

    dataset = VulnerabilityDataset(
        data_path,
        split,
        source_tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
        config,
        ir_tokenizer=ir_tokenizer,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sample_ids = []
    labels = []
    fused_vectors = []
    alpha_means = []

    with torch.no_grad():
        for batch in progress_bar(dataloader, desc=f"Extract B4 {split} features"):
            sample_ids.extend(batch["sample_id"])
            batch = move_batch_to_device(batch, device)
            fused, alpha = b4_fused_vectors(model, batch)
            fused_vectors.append(fused.detach().cpu().numpy())
            alpha_means.extend(alpha.detach().cpu().mean(dim=1).numpy().tolist())
            labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())

    latent_dimension = config["model"].get("projection", {}).get(
        "latent_dimension",
        config["model"]["latent_dimension"],
    )
    fused_array = (
        np.concatenate(fused_vectors, axis=0).astype(np.float32)
        if fused_vectors
        else np.empty((0, latent_dimension), dtype=np.float32)
    )

    output_path = output_dir / f"{split}.npz"
    np.savez_compressed(
        output_path,
        sample_ids=np.asarray(sample_ids, dtype=str),
        labels=np.asarray(labels, dtype=np.int64),
        fused=fused_array,
        alpha_mean=np.asarray(alpha_means, dtype=np.float32),
    )
    print(f"Saved {len(labels)} {split} vectors to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frozen B4 gated-fusion vectors for ablation classifiers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional path to b4_best.pt.")
    parser.add_argument("--output-dir", default=str(FEATURES_DIR))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_ablation_dirs()

    config = load_config(PROJECT_ROOT / args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(config["paths"]["checkpoints"]) / "b4_best.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"B4 checkpoint not found at {checkpoint_path}")

    output_dir = resolve_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or config["training"]["batch_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_tokenizer = AutoTokenizer.from_pretrained(model_source_name(config))
    ir_tokenizer = AutoTokenizer.from_pretrained(model_ir_name(config))
    model = load_b4_model(config, checkpoint_path, device)

    for split in args.splits:
        extract_split(config, model, source_tokenizer, ir_tokenizer, split, output_dir, batch_size, device)


if __name__ == "__main__":
    main()
