"""CLI draft for evaluation."""

from mini_tcn_llm.config import load_yaml, validate_config
from mini_tcn_llm.eval import eval_main


def main(config_path: str = "configs/eval.yaml") -> None:
    cfg = load_yaml(config_path)
    validate_config(cfg)
    _ = eval_main(cfg)


if __name__ == "__main__":
    main()
