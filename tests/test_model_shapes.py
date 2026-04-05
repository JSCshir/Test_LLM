import torch

from mini_tcn_llm.model import build_model


def test_model_forward_shape_and_loss():
    model = build_model(
        {
            "model": {
                "vocab_size": 128,
                "d_model": 32,
                "num_layers": 2,
                "kernel_size": 3,
                "dilations": [1, 2],
                "dropout": 0.0,
                "tie_weights": True,
                "max_length": 16,
            }
        }
    )
    x = torch.randint(0, 128, (4, 16))
    logits, loss = model(x, labels=x)
    assert logits.shape == (4, 16, 128)
    assert loss is not None
    assert loss.item() > 0
