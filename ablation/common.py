import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = ABLATION_DIR / "features"
MODELS_DIR = ABLATION_DIR / "models"
RESULTS_DIR = ABLATION_DIR / "results"

VARIANTS = {
    "b4-a": {
        "name": "B4-A",
        "description": "Adaptive gated fusion + MLP classifier",
        "model_file": "b4-a_mlp.pt",
    },
    "b4-b": {
        "name": "B4-B",
        "description": "Adaptive gated fusion + XGBoost classifier",
        "model_file": "b4-b_xgboost.joblib",
    },
    "b4-c": {
        "name": "B4-C",
        "description": "Adaptive gated fusion + Random Forest classifier",
        "model_file": "b4-c_random_forest.joblib",
    },
}


def ensure_ablation_dirs():
    for directory in [FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_dir(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def feature_path(features_dir, split):
    return resolve_dir(features_dir) / f"{split}.npz"


def model_path(models_dir, variant):
    return resolve_dir(models_dir) / VARIANTS[variant]["model_file"]


def load_features(features_dir, split):
    path = feature_path(features_dir, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split} features at {path}. Run ablation/extract_b4_features.py first."
        )
    data = np.load(path)
    return {
        "sample_ids": data["sample_ids"],
        "labels": data["labels"].astype(int),
        "fused": data["fused"].astype(np.float32),
        "alpha_mean": data["alpha_mean"] if "alpha_mean" in data else None,
    }


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def false_positive_rate(labels, predictions):
    false_positives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 1)
    true_negatives = sum(1 for label, prediction in zip(labels, predictions) if label == 0 and prediction == 0)
    denominator = false_positives + true_negatives
    return false_positives / denominator if denominator else 0.0


def calculate_metrics(labels, probabilities, threshold):
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    roc_auc = roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else 0.0
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc),
        "false_positive_rate": float(false_positive_rate(labels, predictions)),
        "threshold": float(threshold),
    }


def save_results(results_dir, variant, sample_ids, labels, probabilities, threshold):
    results_dir = resolve_dir(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    metrics = {
        "variant": variant,
        "variant_name": VARIANTS[variant]["name"],
        "records": int(len(labels)),
        **calculate_metrics(labels.tolist(), probabilities.tolist(), threshold),
    }

    metrics_path = results_dir / f"{variant}_metrics.json"
    predictions_path = results_dir / f"{variant}_predictions.csv"

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

    return metrics_path, predictions_path, metrics
