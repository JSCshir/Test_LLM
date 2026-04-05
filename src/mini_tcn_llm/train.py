"""Training loop for the mini TCN LLM."""

from pathlib import Path

import torch
from tqdm import tqdm

from mini_tcn_llm.data import create_dataloader, load_token_ids
from mini_tcn_llm.model import build_model
from mini_tcn_llm.utils import seed_everything


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def train_main(config: dict) -> None:
    """Run training loop."""
    seed_everything(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))

    training_cfg = config["training"]
    paths_cfg = config["paths"]
    model_cfg = config["model"]

    train_ids = load_token_ids(paths_cfg["train_tokens"])
    val_ids = load_token_ids(paths_cfg["val_tokens"])

    seq_len = int(model_cfg["max_length"])
    batch_size = int(training_cfg["batch_size"])

    train_loader = create_dataloader(train_ids, seq_len=seq_len, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ids, seq_len=seq_len, batch_size=batch_size, shuffle=False)

    model = build_model({"model": model_cfg}).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    checkpoint_dir = Path(paths_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val = float("inf")

    for epoch in range(int(training_cfg["epochs"])):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, labels=y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip_norm"]))
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": float(loss.item())})

            eval_every = int(training_cfg["eval_every_steps"])
            save_every = int(training_cfg["save_every_steps"])

            if global_step % eval_every == 0:
                val_loss = _evaluate_loss(model, val_loader, device)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict(), "step": global_step}, checkpoint_dir / "best.pt")

            if global_step % save_every == 0:
                torch.save({"model": model.state_dict(), "step": global_step}, checkpoint_dir / "latest.pt")

    torch.save({"model": model.state_dict(), "step": global_step}, checkpoint_dir / "latest.pt")


def _evaluate_loss(model, loader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, labels=y)
            assert loss is not None
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)
