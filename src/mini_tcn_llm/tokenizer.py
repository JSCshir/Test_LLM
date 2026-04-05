"""Tokenizer helpers for a mini LLM pipeline."""

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_tokenizer(corpus_path: str, output_dir: str, vocab_size: int, tokenizer_type: str = "bpe", bert_model_name: str = "bert-base-uncased") -> None:
    """Train and save tokenizer artifacts.

    For `tokenizer_type=bpe`, trains a local BPE tokenizer.
    For `tokenizer_type=bert`, saves a pretrained BERT tokenizer.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer_type == "bert":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "tokenizer_meta.json").write_text(
            json.dumps({"type": "bert", "name": bert_model_name}, indent=2),
            encoding="utf-8",
        )
        return

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "tokenizer_meta.json").write_text(json.dumps({"type": "bpe"}, indent=2), encoding="utf-8")


def load_tokenizer(tokenizer_dir: str) -> Any:
    """Load tokenizer from directory."""
    path = Path(tokenizer_dir)
    meta_path = path / "tokenizer_meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("type") == "bert":
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(path)

    tokenizer_path = path / "tokenizer.json"
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    raise FileNotFoundError(f"No tokenizer artifact found in {tokenizer_dir}")


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Encode text to token ids."""
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text)
        if hasattr(encoded, "ids"):
            return encoded.ids
        if isinstance(encoded, list):
            return encoded

    if hasattr(tokenizer, "__call__"):
        out = tokenizer(text, add_special_tokens=False)
        if isinstance(out, dict) and "input_ids" in out:
            return list(out["input_ids"])

    raise TypeError("Unsupported tokenizer object for encode_text")


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode token ids to text."""
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids)
    raise TypeError("Unsupported tokenizer object for decode_ids")
