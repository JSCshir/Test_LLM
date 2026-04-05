"""CLI for training."""

import argparse

from mini_tcn_llm.config import load_yaml, validate_config
from mini_tcn_llm.train import train_main


def main(config_path: str = "configs/train.yaml", model_config_path: str = "configs/model.yaml") -> None:
    train_cfg = load_yaml(config_path)
    model_cfg = load_yaml(model_config_path)
    validate_config(train_cfg)
    validate_config(model_cfg)

    merged = dict(train_cfg)
    merged["model"] = model_cfg["model"]
    train_main(merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    main(args.config, args.model_config)
