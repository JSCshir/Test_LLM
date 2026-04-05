"""TCN language model implementation."""

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with left padding only (causal)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual TCN block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        residual = x
        y = x.transpose(1, 2)
        y = self.conv1(y).transpose(1, 2)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.dropout(y)

        y = y.transpose(1, 2)
        y = self.conv2(y).transpose(1, 2)
        y = self.norm2(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return residual + y


class TCNLanguageModel(nn.Module):
    """Simple TCN language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        tie_weights: bool,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        if len(dilations) < num_layers:
            raise ValueError("dilations length must be >= num_layers")
        self.blocks = nn.ModuleList(
            [
                TCNBlock(channels=d_model, kernel_size=kernel_size, dilation=dilations[i], dropout=dropout)
                for i in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return logits, loss


def build_model(config: dict):
    """Construct and return the TCN model."""
    if "model" in config:
        cfg = config["model"]
    else:
        cfg = config
    return TCNLanguageModel(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        dilations=[int(d) for d in cfg["dilations"]],
        dropout=float(cfg.get("dropout", 0.0)),
        tie_weights=bool(cfg.get("tie_weights", False)),
    )
