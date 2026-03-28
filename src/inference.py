"""Inférence sur de nouveaux commentaires (FR, EN, ES, AR)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ID2LABEL, LABEL_NAMES, MAX_LENGTH, MODEL_DIR
from .preprocess import clean_text


def load_model(model_path: Path | None = None):
    path = model_path or MODEL_DIR
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    if not path.exists() or not (path / "config.json").exists() or not has_weights:
        raise FileNotFoundError(
            f"Modèle fine-tuné introuvable dans {path}. Entraînez avec: python -m src.train"
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, device: torch.device | None = None) -> tuple[str, float]:
    text = clean_text(text)
    device = device or next(model.parameters()).device
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    conf = float(probs[idx])
    return ID2LABEL.get(idx, LABEL_NAMES[idx]), conf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    p.add_argument("--text", type=str, default="", help="Texte à classifier")
    p.add_argument("--demo", action="store_true", help="Exemples multilingues")
    args = p.parse_args()

    path = Path(args.model_dir)
    tokenizer, model = load_model(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = []
    if args.demo:
        samples = [
            ("en", "This product exceeded my expectations, amazing quality!"),
            ("fr", "Livraison rapide, article conforme, je recommande."),
            ("es", "Pésimo servicio, nunca más compro aquí."),
            ("ar", "الخدمة كانت ممتازة والموظفون لطفاء جداً"),
        ]
    elif args.text:
        samples = [("?", args.text)]
    else:
        print("Utilisez --text \"...\" ou --demo")
        return

    for lang, t in samples:
        label, conf = predict(t, tokenizer, model, device)
        print(f"[{lang}] {label} ({conf:.3f}) | {t[:80]}...")


if __name__ == "__main__":
    main()
