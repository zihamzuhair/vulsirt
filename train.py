"""Train the four CodeBERT-based vulnerability detection baselines."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from helpers.dataset import VulnerabilityDataset
from models import build_model
from helpers.config_loader import ensure_directories, load_config, model_ir_name, model_source_name, primevul_processed_path
from helpers.logger import setup_logger
from helpers.progress import progress_bar


def set_seed(seed):
    """Fix random seeds so repeated runs are easier to compare."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    """Move tensor values in a batch to CPU or GPU, leaving text values as-is."""
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def model_inputs(batch):
    """Keep one common input shape for all baseline models."""
    return {
        "source_input_ids": batch["source_input_ids"],
        "source_attention_mask": batch["source_attention_mask"],
        "ir_input_ids": batch["ir_input_ids"],
        "ir_attention_mask": batch["ir_attention_mask"],
    }


def calculate_metrics(labels, probabilities, threshold):
    """Convert probabilities to labels and calculate basic validation scores."""
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def validate(model, dataloader, criterion, device, threshold):
    """Check the model on validation data without updating weights."""
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


def build_loss(config, device):
    """Create the BCE loss, including optional class weights from config."""
    class_weights = config.get("training", {}).get("class_weights", {})
    negative_weight = float(class_weights.get("non_vulnerable", class_weights.get("negative", 1.0)))
    positive_weight = float(class_weights.get("vulnerable", class_weights.get("positive", 1.0)))

    negative_weight = torch.tensor(negative_weight, dtype=torch.float, device=device)
    positive_weight = torch.tensor(positive_weight, dtype=torch.float, device=device)

    def weighted_bce_loss(logits, labels):
        losses = binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weights = torch.where(labels >= 0.5, positive_weight, negative_weight)
        return (losses * weights).mean()

    return weighted_bce_loss


def checkpoint_paths(config, baseline):
    """Return the best and last checkpoint paths for one baseline."""
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    return checkpoint_dir / f"{baseline}_best.pt", checkpoint_dir / f"{baseline}_last.pt"


def load_resume_checkpoint(model, optimizer, last_path, device):
    """Load the last checkpoint so training can continue from there."""
    checkpoint = torch.load(last_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_f1", 0.0)


def encoder_initialization_path(config, init_config, key, baseline):
    """Resolve the checkpoint used to warm-start a B4 encoder."""
    path = init_config.get(key)
    if path:
        return Path(path)
    return Path(config["paths"]["checkpoints"]) / f"{baseline}_best.pt"


def load_encoder_from_checkpoint(encoder, checkpoint_path, checkpoint_prefix, device):
    """Copy one encoder's weights from a saved baseline checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    prefix = f"{checkpoint_prefix}."
    encoder_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError(f"No {checkpoint_prefix} weights found in {checkpoint_path}")
    encoder.load_state_dict(encoder_state)


def initialize_b4_encoders(model, config, device, logger):
    """Optionally load B1/B2 encoder weights before B4 training starts."""
    init_config = config.get("model", {}).get("encoder_initialization", {})
    if not init_config.get("enabled", False):
        return

    source_checkpoint = encoder_initialization_path(config, init_config, "source_checkpoint", "b1")
    ir_checkpoint = encoder_initialization_path(config, init_config, "ir_checkpoint", "b2")
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"B4 source encoder initialization checkpoint not found: {source_checkpoint}")
    if not ir_checkpoint.exists():
        raise FileNotFoundError(f"B4 IR encoder initialization checkpoint not found: {ir_checkpoint}")

    load_encoder_from_checkpoint(model.source_encoder, source_checkpoint, "source_encoder", device)
    load_encoder_from_checkpoint(model.ir_encoder, ir_checkpoint, "ir_encoder", device)
    logger.info("Initialized B4 source encoder from %s", source_checkpoint)
    logger.info("Initialized B4 IR encoder from %s", ir_checkpoint)


def main():
    """Parse options, build data loaders, train, validate, and save checkpoints."""
    parser = argparse.ArgumentParser(description="Train a CodeBERT vulnerability model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline", choices=["b1", "b2", "b3", "b4"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)
    baseline = (args.baseline or config["training"]["baseline"]).lower()
    logger = setup_logger(f"train_{baseline}", config["paths"]["logs"])

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_tokenizer = AutoTokenizer.from_pretrained(model_source_name(config))
    ir_tokenizer = AutoTokenizer.from_pretrained(model_ir_name(config))

    train_dataset = VulnerabilityDataset(
        primevul_processed_path(config),
        "train",
        source_tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
        config,
        ir_tokenizer=ir_tokenizer,
    )
    validation_dataset = VulnerabilityDataset(
        primevul_processed_path(config),
        "validation",
        source_tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
        config,
        ir_tokenizer=ir_tokenizer,
    )
    logger.info("Loaded %d train and %d validation records", len(train_dataset), len(validation_dataset))

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = build_model(baseline, config).to(device)
    if baseline == "b4":
        initialize_b4_encoders(model, config, device, logger)
    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = build_loss(config, device)
    best_path, last_path = checkpoint_paths(config, baseline)

    start_epoch = 1
    best_f1 = -1.0
    if config["training"].get("resume", True) and last_path.exists():
        start_epoch, best_f1 = load_resume_checkpoint(model, optimizer, last_path, device)
        logger.info("Resumed %s from epoch %d", baseline.upper(), start_epoch)

    training_start_time = time.perf_counter()
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
        training_seconds = round(time.perf_counter() - training_start_time, 3)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        logger.info(
            "Epoch %d train_loss=%.4f val_loss=%.4f val_f1=%.4f val_precision=%.4f val_recall=%.4f training_seconds=%.3f",
            epoch,
            train_loss,
            validation_metrics["loss"],
            validation_metrics["f1"],
            validation_metrics["precision"],
            validation_metrics["recall"],
            training_seconds,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": max(best_f1, validation_metrics["f1"]),
            "training_seconds": training_seconds,
        }
        torch.save(checkpoint, last_path)

        if validation_metrics["f1"] > best_f1:
            best_f1 = validation_metrics["f1"]
            checkpoint["best_f1"] = best_f1
            torch.save(checkpoint, best_path)
            logger.info("Saved new best checkpoint to %s", best_path)


if __name__ == "__main__":
    main()
