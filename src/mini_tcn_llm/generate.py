"""Text generation helpers."""

import torch


def sample_next_token(logits, temperature: float = 1.0, top_k: int | None = None):
    """Sample next token from logits."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.size(-1))
        values, indices = torch.topk(scaled, k=k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        return indices.gather(-1, next_idx)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_tokens(model, input_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int):
    """Autoregressive generation loop."""
    model.eval()
    out = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(out)
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
            out = torch.cat([out, next_token], dim=1)
    return out
