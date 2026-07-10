"""Generate experiment-scoped paths from config hyperparameters."""

import argparse
from pathlib import Path

import yaml

from helpers.config_loader import load_config


DEFAULT_DROPOUT = 0.1
DEFAULT_LATENT_DIMENSION = 256
DEFAULT_TEMPERATURE = 1.0


def format_float(value, decimals=2):
    """Format a float without losing the explicit threshold style."""
    return f"{float(value):.{decimals}f}"


def format_weight(value):
    """Keep class weight names short when the value is a whole number."""
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number).replace(".", "p")


def sample_prefix(config, config_path):
    """Return a stable sample prefix such as 1000_samples."""
    max_records = (
        config.get("data", {})
        .get("datasets", {})
        .get("primevul", {})
        .get("max_records")
    )
    if max_records:
        return f"{max_records}_samples"
    return Path(config_path).stem


def experiment_name(config, config_path):
    """Build the folder name used by results and checkpoint paths."""
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    projection_config = model_config.get("projection", {})
    gating_config = model_config.get("gating", {})
    class_weights = training_config.get("class_weights", {})

    name_parts = [sample_prefix(config, config_path)]

    dropout = float(model_config.get("dropout", DEFAULT_DROPOUT))
    latent_dimension = int(projection_config.get("latent_dimension", DEFAULT_LATENT_DIMENSION))
    temperature = float(gating_config.get("temperature", DEFAULT_TEMPERATURE))
    include_model_params = (
        dropout != DEFAULT_DROPOUT
        or latent_dimension != DEFAULT_LATENT_DIMENSION
        or temperature != DEFAULT_TEMPERATURE
    )
    if include_model_params:
        name_parts.extend(
            [
                "do",
                str(dropout).replace(".", "."),
                "ld",
                str(latent_dimension),
                "temp",
                str(temperature).replace(".", "."),
            ]
        )

    name_parts.extend(
        [
            "th",
            format_float(training_config.get("threshold", 0.5), decimals=2),
            "cw",
            format_weight(class_weights.get("non_vulnerable", 1.0)),
            format_weight(class_weights.get("vulnerable", 1.0)),
            "ep",
            str(training_config.get("epochs")),
        ]
    )
    return "_".join(name_parts)


def checkpoint_root(config):
    """Use the existing checkpoint parent as the experiment root."""
    checkpoint_path = Path(config["paths"]["checkpoints"])
    if checkpoint_path.name:
        return checkpoint_path.parent
    return checkpoint_path


def update_experiment_paths(config, config_path):
    """Return a copied config with result/checkpoint/log paths set by experiment name."""
    updated = dict(config)
    updated["paths"] = dict(config["paths"])
    name = experiment_name(config, config_path)
    updated["paths"]["results"] = str(Path("results") / name)
    updated["paths"]["checkpoints"] = str(checkpoint_root(config) / name)
    updated["paths"]["logs"] = str(Path("logs") / name)
    return name, updated


def write_generated_config(config_path, output_dir):
    """Write a generated config and return its path."""
    config = load_config(config_path)
    name, updated = update_experiment_paths(config, config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.yaml"
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(updated, file, sort_keys=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate experiment-scoped config paths.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="temp/generated_configs")
    args = parser.parse_args()

    print(write_generated_config(args.config, args.output_dir))


if __name__ == "__main__":
    main()
