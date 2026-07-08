import argparse

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

from common import (
    FEATURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    VARIANTS,
    calculate_metrics,
    ensure_ablation_dirs,
    load_features,
    model_path,
    resolve_dir,
)

from helpers.config_loader import load_config


class MLPClassifier(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp_probabilities(model, features, device, batch_size=256):
    model.eval()
    probabilities = []
    tensor = torch.tensor(features, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_features,) in loader:
            logits = model(batch_features.to(device))
            probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=np.float32)


def train_mlp(config, features_dir, models_dir, args):
    train_data = load_features(features_dir, "train")
    validation_data = load_features(features_dir, "validation")
    if len(train_data["labels"]) == 0:
        raise ValueError("Cannot train B4-A: train.npz has no records.")

    seed = config["training"]["seed"]
    threshold = config["training"]["threshold"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dimension = train_data["fused"].shape[1]
    model = MLPClassifier(input_dimension, args.hidden_size, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.tensor(train_data["fused"], dtype=torch.float32),
        torch.tensor(train_data["labels"], dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    best_f1 = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        probabilities = mlp_probabilities(model, validation_data["fused"], device)
        metrics = calculate_metrics(validation_data["labels"].tolist(), probabilities.tolist(), threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dimension": input_dimension,
                "hidden_size": args.hidden_size,
                "dropout": args.dropout,
                "threshold": threshold,
                "epoch": epoch,
                "validation_f1": best_f1,
            }
        print(
            f"Epoch {epoch:03d} loss={np.mean(epoch_losses):.4f} "
            f"val_f1={metrics['f1']:.4f} val_precision={metrics['precision']:.4f} val_recall={metrics['recall']:.4f}"
        )

    output_path = model_path(models_dir, "b4-a")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)
    print(f"Saved B4-A MLP head to {output_path}")


def train_xgboost(config, features_dir, models_dir):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for B4-B. Install it with: pip install -r ablation/requirements-ablation.txt"
        ) from exc

    train_data = load_features(features_dir, "train")
    validation_data = load_features(features_dir, "validation")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=config["training"]["seed"],
        n_jobs=-1,
    )
    model.fit(
        train_data["fused"],
        train_data["labels"],
        eval_set=[(validation_data["fused"], validation_data["labels"])],
        verbose=False,
    )

    output_path = model_path(models_dir, "b4-b")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved B4-B XGBoost head to {output_path}")


def train_random_forest(config, features_dir, models_dir):
    train_data = load_features(features_dir, "train")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config["training"]["seed"],
        n_jobs=-1,
    )
    model.fit(train_data["fused"], train_data["labels"])

    output_path = model_path(models_dir, "b4-c")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved B4-C Random Forest head to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train B4 ablation classifier heads on frozen fused vectors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=list(VARIANTS.keys()) + ["all"], required=True)
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_ablation_dirs()
    config = load_config(PROJECT_ROOT / args.config)
    features_dir = resolve_dir(args.features_dir)
    models_dir = resolve_dir(args.models_dir)

    variants = ["b4-a", "b4-b", "b4-c"] if args.variant == "all" else [args.variant]
    for variant in variants:
        print(f"Training {VARIANTS[variant]['name']}: {VARIANTS[variant]['description']}")
        if variant == "b4-a":
            train_mlp(config, features_dir, models_dir, args)
        elif variant == "b4-b":
            train_xgboost(config, features_dir, models_dir)
        elif variant == "b4-c":
            train_random_forest(config, features_dir, models_dir)


if __name__ == "__main__":
    main()
