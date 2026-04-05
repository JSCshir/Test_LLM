"""Small utility script to inspect BERT embeddings for input text."""

import argparse
import json

import torch


def main(text: str, model_name: str = "bert-base-uncased") -> None:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**batch)

    payload = {
        "model": model_name,
        "input_text": text,
        "input_shape": list(batch["input_ids"].shape),
        "last_hidden_state_shape": list(out.last_hidden_state.shape),
        "cls_embedding_first8": out.last_hidden_state[0, 0, :8].tolist(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="bert-base-uncased")
    args = parser.parse_args()
    main(text=args.text, model_name=args.model)
