import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import VulnerabilityDataset
from models import build_model
from train import model_inputs, move_batch_to_device
from utils.config import ensure_directories, load_config, primevul_processed_path, rust_processed_path
from utils.logger import setup_logger
from utils.progress import progress_bar


def false_positive_rate(labels, predictions):
    false_positives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    true_negatives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)
    denominator = false_positives + true_negatives
    return false_positives / denominator if denominator else 0.0


def evaluate_baseline(baseline, config, device, logger, data_path=None, split="test", dataset_name="primevul"):
    checkpoint_path = Path(config["paths"]["checkpoints"]) / f"{baseline}_best.pt"
    if not checkpoint_path.exists():
        logger.info("Skipping %s: checkpoint not found at %s", baseline.upper(), checkpoint_path)
        return None
    data_path = data_path or primevul_processed_path(config)
    if not Path(data_path).exists():
        logger.info("Skipping %s on %s: processed data not found at %s", baseline.upper(), dataset_name, data_path)
        return None

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    dataset = VulnerabilityDataset(
        data_path,
        split,
        tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
        config,
    )
    if len(dataset) == 0:
        logger.info("Skipping %s on %s: no records for split %s", baseline.upper(), dataset_name, split)
        return None
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(baseline, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = []
    probabilities = []
    sample_ids = []

    with torch.no_grad():
        for batch in progress_bar(dataloader, desc=f"Evaluate {baseline.upper()} on {dataset_name}"):
            batch = move_batch_to_device(batch, device)
            output = model(**model_inputs(batch))
            probabilities.extend(torch.sigmoid(output["logits"]).detach().cpu().numpy().tolist())
            labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
            sample_ids.extend(batch["sample_id"])

    threshold = config["training"]["threshold"]
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    roc_auc = roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else 0.0

    metrics = {
        "baseline": baseline,
        "dataset": dataset_name,
        "records": len(dataset),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc,
        "false_positive_rate": false_positive_rate(labels, predictions),
        "threshold": threshold,
    }

    results_dir = Path(config["paths"]["results"])
    suffix = "" if dataset_name == "primevul" else f"_{dataset_name}"
    metrics_path = results_dir / f"{baseline}{suffix}_metrics.json"
    predictions_path = results_dir / f"{baseline}{suffix}_predictions.csv"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label": labels,
            "probability": probabilities,
            "prediction": predictions,
        }
    ).to_csv(predictions_path, index=False)

    logger.info("Saved %s %s metrics and predictions", baseline.upper(), dataset_name)
    return metrics


def evaluation_targets(config, selected_dataset):
    paths = config["paths"]
    targets = []
    if selected_dataset in {"primevul", "all"}:
        targets.append(
            {
                "name": "primevul",
                "path": primevul_processed_path(config),
                "split": "test",
            }
        )
    rust_path = rust_processed_path(config)
    if selected_dataset in {"rust", "all"} and rust_path:
        targets.append(
            {
                "name": "rust",
                "path": rust_path,
                "split": "test",
            }
        )
    return targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained baselines.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline", choices=["b1", "b2", "b3", "b4"], default=None)
    parser.add_argument("--dataset", choices=["primevul", "rust", "all"], default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    logger = setup_logger("evaluate", config["paths"]["logs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baselines = [args.baseline] if args.baseline else ["b1", "b2", "b3", "b4"]
    all_metrics = []
    for baseline in baselines:
        for target in evaluation_targets(config, args.dataset):
            metrics = evaluate_baseline(
                baseline,
                config,
                device,
                logger,
                data_path=target["path"],
                split=target["split"],
                dataset_name=target["name"],
            )
            if metrics is not None:
                all_metrics.append(metrics)

    if all_metrics:
        comparison_path = Path(config["paths"]["results"]) / "baseline_comparison.csv"
        pd.DataFrame(all_metrics).to_csv(comparison_path, index=False)
        logger.info("Saved baseline comparison to %s", comparison_path)


if __name__ == "__main__":
    main()
