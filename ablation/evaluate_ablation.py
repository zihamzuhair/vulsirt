"""Evaluate trained B4 ablation classifier heads."""

import argparse

import joblib
import torch

from common import (
    FEATURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    VARIANTS,
    ensure_ablation_dirs,
    load_features,
    model_path,
    resolve_dir,
    save_results,
    sigmoid,
)
from train_ablation import MLPClassifier, mlp_probabilities

from helpers.config_loader import load_config


def evaluate_mlp(features, models_dir, threshold):
    """Load the B4-A MLP head and return its probabilities."""
    checkpoint_path = model_path(models_dir, "b4-a")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing B4-A model at {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MLPClassifier(
        checkpoint["input_dimension"],
        checkpoint["hidden_size"],
        checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    probabilities = mlp_probabilities(model, features["fused"], device)
    return probabilities, checkpoint.get("threshold", threshold)


def evaluate_sklearn_classifier(features, models_dir, variant):
    """Load a sklearn-style ablation model and return probabilities."""
    checkpoint_path = model_path(models_dir, variant)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing {VARIANTS[variant]['name']} model at {checkpoint_path}")
    model = joblib.load(checkpoint_path)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features["fused"])[:, 1]
    return sigmoid(model.decision_function(features["fused"]))


def parse_args():
    """Read ablation evaluation options from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate B4 ablation classifier heads.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=list(VARIANTS.keys()) + ["all"], required=True)
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main():
    """Evaluate one or all ablation variants and save result files."""
    args = parse_args()
    ensure_ablation_dirs()
    config = load_config(PROJECT_ROOT / args.config)
    threshold = args.threshold if args.threshold is not None else config["training"]["threshold"]
    features_dir = resolve_dir(args.features_dir)
    models_dir = resolve_dir(args.models_dir)
    results_dir = resolve_dir(args.results_dir)
    features = load_features(features_dir, args.split)

    variants = ["b4-a", "b4-b", "b4-c"] if args.variant == "all" else [args.variant]
    for variant in variants:
        if variant == "b4-a":
            probabilities, variant_threshold = evaluate_mlp(features, models_dir, threshold)
        else:
            probabilities = evaluate_sklearn_classifier(features, models_dir, variant)
            variant_threshold = threshold

        metrics_path, predictions_path, metrics = save_results(
            results_dir,
            variant,
            features["sample_ids"],
            features["labels"],
            probabilities,
            variant_threshold,
        )
        print(
            f"{VARIANTS[variant]['name']} f1={metrics['f1']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
        )
        print(f"Saved metrics to {metrics_path}")
        print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
