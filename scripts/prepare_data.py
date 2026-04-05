"""CLI draft for data preparation."""

from mini_tcn_llm.config import load_yaml, validate_config


def main(config_path: str = "configs/data.yaml") -> None:
    cfg = load_yaml(config_path)
    validate_config(cfg)
    raise NotImplementedError("Implement data cleaning/tokenization pipeline")


if __name__ == "__main__":
    main()
