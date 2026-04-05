"""Tokenizer helpers for a mini LLM pipeline."""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def train_tokenizer(corpus_path: str, output_dir: str, vocab_size: int) -> None:
    """Train and save tokenizer artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    tokenizer.save(str(out_dir / "tokenizer.json"))


def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load tokenizer from directory."""
    path = Path(tokenizer_dir) / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode text to token ids."""
    return tokenizer.encode(text).ids


def decode_ids(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    """Decode token ids to text."""
    return tokenizer.decode(token_ids)
