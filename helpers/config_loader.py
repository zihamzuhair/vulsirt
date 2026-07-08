"""Load YAML config files and resolve common project paths."""

from pathlib import Path

import yaml


def deep_merge(base, override):
    """Merge nested config dictionaries, keeping base values unless overridden."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path="configs/config.yaml"):
    """Load a config file, including any parent config it inherits."""
    return _load_config(Path(config_path), seen=set())


def _load_config(config_path, seen):
    """Load one config file and guard against circular inheritance."""
    config_path = config_path.resolve()
    if config_path in seen:
        raise ValueError(f"Circular config inheritance detected at {config_path}")
    seen.add(config_path)

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    base_path = config.pop("inherits", config.pop("extends", None))
    if not base_path:
        return config

    base_path = Path(base_path)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path

    base_config = _load_config(base_path, seen)
    return deep_merge(base_config, config)


def ensure_directories(config):
    """Create output folders needed for checkpoints, results, logs, and data."""
    for key in ["checkpoints", "results", "logs"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    for key in ["processed_primevul", "processed_rust", "processed_data"]:
        if key in config.get("paths", {}):
            Path(config["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
    if "llvm_errors" in config.get("paths", {}):
        Path(config["paths"]["llvm_errors"]).parent.mkdir(parents=True, exist_ok=True)


def primevul_processed_path(config):
    """Return the processed PrimeVul path using old or new config keys."""
    paths = config["paths"]
    return paths.get("processed_primevul", paths.get("processed_data"))


def rust_processed_path(config):
    """Return the processed Rust path when the config defines it."""
    return config["paths"].get("processed_rust")


def model_source_name(config):
    """Return the Hugging Face model name used for source code."""
    model_config = config.get("model", {})
    return model_config.get("source_name")


def model_ir_name(config):
    """Return the Hugging Face model name used for LLVM IR."""
    model_config = config.get("model", {})
    return model_config.get("ir_name")
