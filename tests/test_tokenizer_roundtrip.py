from pathlib import Path

from mini_tcn_llm.tokenizer import decode_ids, encode_text, load_tokenizer, train_tokenizer


def test_tokenizer_roundtrip(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("In the beginning God created the heaven and the earth.", encoding="utf-8")

    out_dir = tmp_path / "tok"
    train_tokenizer(str(corpus), str(out_dir), 128)
    tokenizer = load_tokenizer(str(out_dir))

    text = "God created the earth"
    ids = encode_text(tokenizer, text)
    decoded = decode_ids(tokenizer, ids)

    assert len(ids) > 0
    assert isinstance(decoded, str)
    assert "God" in decoded or "god" in decoded
