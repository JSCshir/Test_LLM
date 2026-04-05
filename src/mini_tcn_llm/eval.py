"""Evaluation for the mini TCN LLM."""

import math

import torch

from mini_tcn_llm.data import create_dataloader, load_token_ids
from mini_tcn_llm.generate import generate_tokens
from mini_tcn_llm.model import build_model
from mini_tcn_llm.tokenizer import decode_ids, encode_text, load_tokenizer


def eval_main(config: dict) -> dict:
    """Evaluate a checkpoint."""
    model_cfg = config["model"]
    model = build_model({"model": model_cfg})

    ckpt = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ids = load_token_ids(config["val_tokens"])
    loader = create_dataloader(val_ids, seq_len=int(model_cfg["max_length"]), batch_size=int(config["batch_size"]), shuffle=False)

    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            _, loss = model(x, labels=y)
            assert loss is not None
            losses.append(float(loss.item()))

    mean_loss = sum(losses) / max(len(losses), 1)
    metrics = {"loss": mean_loss, "perplexity": math.exp(mean_loss)}

    generation_cfg = config.get("generation", {})
    tokenizer = load_tokenizer(config.get("tokenizer_dir", "data/tokenizer"))
    prompt_ids = encode_text(tokenizer, generation_cfg.get("prompt", "In the beginning"))
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    sampled = generate_tokens(
        model,
        prompt,
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 40)),
        temperature=float(generation_cfg.get("temperature", 1.0)),
        top_k=int(generation_cfg.get("top_k", 40)),
    )
    metrics["sample_text"] = decode_ids(tokenizer, sampled[0].tolist())
    return metrics
