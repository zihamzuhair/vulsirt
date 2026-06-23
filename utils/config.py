from pathlib import Path

import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_directories(config):
    for key in ["checkpoints", "results", "logs"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["processed_data"]).parent.mkdir(parents=True, exist_ok=True)
