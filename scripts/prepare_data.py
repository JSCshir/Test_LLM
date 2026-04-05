"""CLI for data preparation."""

import argparse
from pathlib import Path

from mini_tcn_llm.config import load_yaml, validate_config
from mini_tcn_llm.data import clean_text, extract_text_from_pdf, save_token_ids
from mini_tcn_llm.tokenizer import encode_text, load_tokenizer, train_tokenizer


def main(config_path: str = "configs/data.yaml") -> None:
    cfg = load_yaml(config_path)
    validate_config(cfg)

    raw_path = cfg["raw_corpus_path"]
    processed_dir = Path(cfg["processed_dir"])
    tokenizer_dir = Path(cfg["tokenizer_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if raw_path.lower().endswith(".pdf"):
        raw_text = extract_text_from_pdf(raw_path)
        corpus_path = processed_dir / "corpus.txt"
        corpus_path.write_text(raw_text, encoding="utf-8")
    else:
        corpus_path = Path(raw_path)
        raw_text = corpus_path.read_text(encoding="utf-8")

    cleaned = clean_text(
        raw_text,
        lowercase=bool(cfg.get("cleaning", {}).get("lowercase", False)),
        strip_extra_whitespace=bool(cfg.get("cleaning", {}).get("strip_extra_whitespace", True)),
    )

    cleaned_path = processed_dir / "cleaned_corpus.txt"
    cleaned_path.write_text(cleaned, encoding="utf-8")

    train_tokenizer(
        corpus_path=str(cleaned_path),
        output_dir=str(tokenizer_dir),
        vocab_size=int(cfg.get("tokenizer", {}).get("vocab_size", 8192)),
    )

    tokenizer = load_tokenizer(str(tokenizer_dir))
    token_ids = encode_text(tokenizer, cleaned)

    train_ratio = float(cfg.get("split", {}).get("train_ratio", 0.98))
    split_idx = int(len(token_ids) * train_ratio)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    save_token_ids(train_ids, str(processed_dir / "train_tokens.bin"))
    save_token_ids(val_ids, str(processed_dir / "val_tokens.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    main(args.config)
