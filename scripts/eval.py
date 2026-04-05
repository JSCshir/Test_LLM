"""CLI for evaluation."""

import argparse
import json

from mini_tcn_llm.config import load_yaml, validate_config
from mini_tcn_llm.eval import eval_main


def main(config_path: str = "configs/eval.yaml", model_config_path: str = "configs/model.yaml") -> None:
    eval_cfg = load_yaml(config_path)
    model_cfg = load_yaml(model_config_path)
    validate_config(eval_cfg)
    validate_config(model_cfg)

    merged = dict(eval_cfg)
    merged["model"] = model_cfg["model"]
    metrics = eval_main(merged)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()
    main(args.config, args.model_config)
