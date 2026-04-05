import pytest

from mini_tcn_llm.model import build_model


def test_model_placeholder_not_implemented():
    with pytest.raises(NotImplementedError):
        build_model({})
