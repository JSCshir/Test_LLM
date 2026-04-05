from mini_tcn_llm.data import build_token_windows


def test_build_token_windows_stride_2():
    token_ids = list(range(10))
    windows = build_token_windows(token_ids, seq_len=4, stride=2)
    assert windows == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]
