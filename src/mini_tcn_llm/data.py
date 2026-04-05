"""Dataset preparation utilities.

Draft module: implement sequence chunking + DataLoader creation.
"""


def build_token_windows(token_ids: list[int], seq_len: int, stride: int) -> list[list[int]]:
    """Create fixed-length token windows from a flat token sequence."""
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive")
    return [token_ids[i : i + seq_len] for i in range(0, len(token_ids) - seq_len + 1, stride)]
