"""General utilities for reproducibility and filesystem helpers."""

import random


def seed_everything(seed: int) -> None:
    """Seed random generators (draft)."""
    random.seed(seed)
