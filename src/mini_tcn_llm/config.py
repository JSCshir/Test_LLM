"""Configuration loading and validation.

Draft module: replace TODOs with concrete dataclasses/pydantic models.
"""

from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    """Minimal placeholder validation.

    TODO: enforce schema and strict type checks.
    """
    if not isinstance(cfg, dict):
        raise TypeError("Config must be a dictionary")
