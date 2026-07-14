"""Train the three simple classifier ablations on the same frozen B4 backbone."""

import argparse
import copy
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.gated_fusion_mlp import GatedFusionMLP
from ablation.gated_fusion_random_forest import GatedFusionRandomForest
from ablation.gated_fusion_xgboost import GatedFusionXGBoost
from helpers.config_loader import load_config, model_ir_name, model_source_name, primevul_processed_path
from helpers.dataset import VulnerabilityDataset


MODEL_FILES = {
    "mlp": "gated_fusion_mlp.pt",
    "xgboost": "gated_fusion_xgboost.joblib",
    "random-forest": "gated_fusion_random_forest.joblib",
}


def make_model(variant, config):
    model_class = {
        "mlp": GatedFusionMLP,
        "xgboost": GatedFusionXGBoost,
        "random-forest": GatedFusionRandomForest,
    }[variant]
    return model_class(
        model_source_name(config),
        model_ir_name(config),
        config["model"].get("projection", {}).get("latent_dimension", 256),
        config["model"]["dropout"],
        config["model"].get("gating", {}),
    )


def make_loader(config, split, source_tokenizer, ir_tokenizer, shuffle=False):
    data_path = Path(primevul_processed_path(config))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    dataset = VulnerabilityDataset(
        data_path,
        split,
        source_tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
        config,
        ir_tokenizer=ir_tokenizer,
    )
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle)


def extract_features(model, loader, device):
    """Run the frozen B4 backbone and keep its fused vectors in memory."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch in loader:
            fused, _ = model.fused_features(
                batch["source_input_ids"].to(device),
                batch["source_attention_mask"].to(device),
                batch["ir_input_ids"].to(device),
                batch["ir_attention_mask"].to(device),
            )
            features.append(fused.cpu().numpy())
            labels.extend(batch["label"].numpy().astype(int).tolist())
    return np.concatenate(features), np.asarray(labels)


def f1_score(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    true_positives = np.sum((labels == 1) & predictions)
    false_positives = np.sum((labels == 0) & predictions)
    false_negatives = np.sum((labels == 1) & ~predictions)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0


def train_mlp(model, train_features, train_labels, validation_features, validation_labels, args, config, device):
    """Train only the MLP and keep the epoch with the best validation F1."""
    model.freeze_backbone()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.classifier_batch_size, shuffle=True)
    validation_tensor = torch.tensor(validation_features, dtype=torch.float32, device=device)

    best_f1, best_state = -1, None
    for epoch in range(1, args.epochs + 1):
        model.classifier.train()
        losses = []
        for features, labels in loader:
            optimizer.zero_grad()
            logits = model.classifier(features.to(device)).squeeze(-1)
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.classifier.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model.classifier(validation_tensor).squeeze(-1)).cpu().numpy()
        score = f1_score(validation_labels, probabilities, config["training"]["threshold"])
        if score > best_f1:
            best_f1 = score
            best_state = copy.deepcopy(model.classifier.state_dict())
        print(f"Epoch {epoch:02d}: loss={np.mean(losses):.4f}, validation_f1={score:.4f}")
    return {"classifier_state_dict": best_state}


def main():
    parser = argparse.ArgumentParser(description="Train B4 classifier ablations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=["mlp", "xgboost", "random-forest", "all"], required=True)
    parser.add_argument("--checkpoint", help="Path to b4_best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(config["paths"]["checkpoints"]) / "b4_best.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    b4_checkpoint = torch.load(checkpoint_path, map_location=device)

    source_tokenizer = AutoTokenizer.from_pretrained(model_source_name(config))
    ir_tokenizer = AutoTokenizer.from_pretrained(model_ir_name(config))
    train_loader = make_loader(config, "train", source_tokenizer, ir_tokenizer)
    validation_loader = make_loader(config, "validation", source_tokenizer, ir_tokenizer)
    output_dir = Path(__file__).resolve().parent / "models"
    output_dir.mkdir(exist_ok=True)

    variants = ["mlp", "xgboost", "random-forest"] if args.variant == "all" else [args.variant]
    for variant in variants:
        print(f"\nTraining {variant.upper()}")
        model = make_model(variant, config).to(device)
        model.load_b4_backbone(b4_checkpoint)
        train_features, train_labels = extract_features(model, train_loader, device)

        if variant == "mlp":
            validation_features, validation_labels = extract_features(model, validation_loader, device)
            saved_model = train_mlp(
                model, train_features, train_labels, validation_features, validation_labels, args, config, device
            )
            torch.save(saved_model, output_dir / MODEL_FILES[variant])
        else:
            model.classifier.set_params(random_state=config["training"]["seed"])
            model.classifier.fit(train_features, train_labels)
            joblib.dump(model.classifier, output_dir / MODEL_FILES[variant])

        print(f"Saved {output_dir / MODEL_FILES[variant]}")


if __name__ == "__main__":
    main()
