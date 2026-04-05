import pytest

from mini_tcn_llm.tokenizer import train_tokenizer


def test_tokenizer_placeholder_not_implemented():
    with pytest.raises(NotImplementedError):
        train_tokenizer("data/raw/corpus.txt", "data/tokenizer", 1024)
