from pathlib import Path

import yaml


def deep_merge(base, override):
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
    return _load_config(Path(config_path), seen=set())


def _load_config(config_path, seen):
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
    for key in ["checkpoints", "results", "logs"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["processed_data"]).parent.mkdir(parents=True, exist_ok=True)
