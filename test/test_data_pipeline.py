import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import VulnerabilityDataset
from helpers.config import load_config, model_ir_name, model_source_name


def summarize_split(split_name, config, source_tokenizer, ir_tokenizer):
    dataset = VulnerabilityDataset(
        config["paths"]["processed_data"],
        split_name,
        source_tokenizer,
        config["model"]["source_max_length"],
        config["model"]["ir_max_length"],
        config,
        ir_tokenizer=ir_tokenizer,
    )
    print(f"{split_name}: {len(dataset)} records")

    if len(dataset) == 0:
        return

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    batch = next(iter(loader))
    print(f"  source_input_ids: {tuple(batch['source_input_ids'].shape)}")
    print(f"  ir_input_ids: {tuple(batch['ir_input_ids'].shape)}")
    print(f"  labels: {batch['label'].tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Test preprocessing output and dataset loading.")
    parser.add_argument("--config", default="test/test_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    source_tokenizer = AutoTokenizer.from_pretrained(model_source_name(config))
    ir_tokenizer = AutoTokenizer.from_pretrained(model_ir_name(config))

    summarize_split("train", config, source_tokenizer, ir_tokenizer)
    summarize_split("validation", config, source_tokenizer, ir_tokenizer)
    summarize_split("test", config, source_tokenizer, ir_tokenizer)
    print("Dataset loading test completed.")


if __name__ == "__main__":
    main()
