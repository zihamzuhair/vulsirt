import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import VulnerabilityDataset
from models import build_model
from utils.config import ensure_directories, load_config
from utils.logger import setup_logger
from utils.progress import progress_bar


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def model_inputs(batch):
    return {
        "source_input_ids": batch["source_input_ids"],
        "source_attention_mask": batch["source_attention_mask"],
        "ir_input_ids": batch["ir_input_ids"],
        "ir_attention_mask": batch["ir_attention_mask"],
    }


def calculate_metrics(labels, probabilities, threshold):
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def validate(model, dataloader, criterion, device, threshold):
    model.eval()
    losses = []
    labels = []
    probabilities = []

    with torch.no_grad():
        for batch in progress_bar(dataloader, desc="Validation"):
            batch = move_batch_to_device(batch, device)
            output = model(**model_inputs(batch))
            loss = criterion(output["logits"], batch["label"])
            losses.append(loss.item())
            probs = torch.sigmoid(output["logits"]).detach().cpu().numpy().tolist()
            probabilities.extend(probs)
            labels.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())

    metrics = calculate_metrics(labels, probabilities, threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def checkpoint_paths(config, baseline):
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    return checkpoint_dir / f"{baseline}_best.pt", checkpoint_dir / f"{baseline}_last.pt"


def load_resume_checkpoint(model, optimizer, last_path, device):
    checkpoint = torch.load(last_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_f1", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train a CodeBERT vulnerability model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--baseline", choices=["b1", "b2", "b3", "b4"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    baseline = (args.baseline or config["training"]["baseline"]).lower()
    logger = setup_logger(f"train_{baseline}", config["paths"]["logs"])

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    train_dataset = VulnerabilityDataset(
        config["paths"]["processed_data"],
        "train",
        tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
    )
    validation_dataset = VulnerabilityDataset(
        config["paths"]["processed_data"],
        "validation",
        tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(baseline, config).to(device)
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = BCEWithLogitsLoss()
    best_path, last_path = checkpoint_paths(config, baseline)

    start_epoch = 1
    best_f1 = -1.0
    if config["training"].get("resume", True) and last_path.exists():
        start_epoch, best_f1 = load_resume_checkpoint(model, optimizer, last_path, device)
        logger.info("Resumed %s from epoch %d", baseline.upper(), start_epoch)

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()
        train_losses = []
        progress = progress_bar(train_loader, desc=f"Epoch {epoch}")

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            output = model(**model_inputs(batch))
            loss = criterion(output["logits"], batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("gradient_clip", 1.0))
            optimizer.step()
            train_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation_metrics = validate(
            model,
            validation_loader,
            criterion,
            device,
            config["training"]["threshold"],
        )
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        logger.info(
            "Epoch %d train_loss=%.4f val_loss=%.4f val_f1=%.4f val_precision=%.4f val_recall=%.4f",
            epoch,
            train_loss,
            validation_metrics["loss"],
            validation_metrics["f1"],
            validation_metrics["precision"],
            validation_metrics["recall"],
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": max(best_f1, validation_metrics["f1"]),
        }
        torch.save(checkpoint, last_path)

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            checkpoint["best_f1"] = best_f1
            torch.save(checkpoint, best_path)
            logger.info("Saved new best checkpoint to %s", best_path)


if __name__ == "__main__":
    main()
