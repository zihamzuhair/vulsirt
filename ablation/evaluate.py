"""Evaluate one or all of the three simple B4 classifier ablations."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
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


def false_positive_rate(labels, predictions):
    false_positives = np.sum((labels == 0) & (predictions == 1))
    true_negatives = np.sum((labels == 0) & (predictions == 0))
    return float(false_positives / (false_positives + true_negatives)) if false_positives + true_negatives else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate B4 classifier ablations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=["mlp", "xgboost", "random-forest", "all"], required=True)
    parser.add_argument("--checkpoint", help="Path to b4_best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = args.threshold if args.threshold is not None else config["training"]["threshold"]

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(config["paths"]["checkpoints"]) / "b4_best.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    b4_checkpoint = torch.load(checkpoint_path, map_location=device)

    data_path = Path(primevul_processed_path(config))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    source_tokenizer = AutoTokenizer.from_pretrained(model_source_name(config))
    ir_tokenizer = AutoTokenizer.from_pretrained(model_ir_name(config))
    dataset = VulnerabilityDataset(
        data_path,
        args.split,
        source_tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
        config,
        ir_tokenizer=ir_tokenizer,
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    models_dir = Path(__file__).resolve().parent / "models"
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    variants = ["mlp", "xgboost", "random-forest"] if args.variant == "all" else [args.variant]

    for variant in variants:
        model = make_model(variant, config).to(device)
        model.load_b4_backbone(b4_checkpoint)
        model.eval()

        if variant == "mlp":
            saved = torch.load(models_dir / MODEL_FILES[variant], map_location=device)
            model.classifier.load_state_dict(saved["classifier_state_dict"])
        else:
            model.classifier = joblib.load(models_dir / MODEL_FILES[variant])

        features, labels, sample_ids = [], [], []
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
                sample_ids.extend(batch["sample_id"])

        features = np.concatenate(features)
        labels = np.asarray(labels)
        if variant == "mlp":
            with torch.no_grad():
                tensor = torch.tensor(features, dtype=torch.float32, device=device)
                probabilities = torch.sigmoid(model.classifier(tensor).squeeze(-1)).cpu().numpy()
        else:
            probabilities = model.classifier.predict_proba(features)[:, 1]

        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "variant": variant,
            "records": len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else 0.0,
            "false_positive_rate": false_positive_rate(labels, predictions),
            "threshold": threshold,
        }

        with open(results_dir / f"{variant}_metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        pd.DataFrame(
            {"sample_id": sample_ids, "label": labels, "probability": probabilities, "prediction": predictions}
        ).to_csv(results_dir / f"{variant}_predictions.csv", index=False)
        print(f"{variant.upper()}: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
