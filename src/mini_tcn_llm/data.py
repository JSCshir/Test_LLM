"""Dataset preparation utilities."""

from pathlib import Path
import re

import numpy as np
import torch
from pypdf import PdfReader
from torch.utils.data import DataLoader, Dataset


class TokenSequenceDataset(Dataset):
    """Language modeling dataset over contiguous token ids."""

    def __init__(self, token_ids: np.ndarray, seq_len: int):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1D array")
        if len(token_ids) <= seq_len:
            raise ValueError("token_ids length must be greater than seq_len")
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.token_ids[idx : idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.token_ids[idx + 1 : idx + self.seq_len + 1].astype(np.int64))
        return x, y


def build_token_windows(token_ids: list[int], seq_len: int, stride: int) -> list[list[int]]:
    """Create fixed-length token windows from a flat token sequence."""
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive")
    return [token_ids[i : i + seq_len] for i in range(0, len(token_ids) - seq_len + 1, stride)]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF."""
    reader = PdfReader(pdf_path)
    chunks: list[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def clean_text(text: str, lowercase: bool, strip_extra_whitespace: bool) -> str:
    """Run lightweight text cleanup steps."""
    if lowercase:
        text = text.lower()
    if strip_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def save_token_ids(token_ids: list[int], output_path: str) -> None:
    """Save token ids as uint32 numpy binary."""
    arr = np.asarray(token_ids, dtype=np.uint32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_path)


def load_token_ids(path: str) -> np.ndarray:
    """Load uint32 token ids from binary file."""
    return np.fromfile(path, dtype=np.uint32)


def create_dataloader(token_ids: np.ndarray, seq_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    """Create torch dataloader from raw token ids."""
    dataset = TokenSequenceDataset(token_ids=token_ids, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
