"""CLI draft for training."""

from mini_tcn_llm.config import load_yaml, validate_config
from mini_tcn_llm.train import train_main


def main(config_path: str = "configs/train.yaml") -> None:
    cfg = load_yaml(config_path)
    validate_config(cfg)
    train_main(cfg)


if __name__ == "__main__":
    main()
