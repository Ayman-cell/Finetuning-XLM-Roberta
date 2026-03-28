#!/usr/bin/env python3
"""
Construction d'un jeu de données RÉEL multilingue (FR, EN, ES, AR, DARIJA).
Classe NEUTRAL supprimée pour meilleur équilibre du dataset.

Sources (Hugging Face, données publiques) :
- Anglais, français, espagnol : avis Amazon multilingues (mteb/amazon_reviews_multi),
  étoiles agrégées en sentiment 2 classes (négatif / positif).
- Arabe & Darija : datasets de tweets sentiment avec filtrage des neutres.

Sortie : data/processed/dataset.parquet (+ CSV optionnel).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

# Racine projet (dossier sentiment_multilingual) : nécessaire pour `from src.…`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_text, is_valid_sample  # noqa: E402

AMAZON_JSONL_BASE = (
    "https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/main"
)
AMAZON_LANGS = ("en", "fr", "es")
# Arabe : binaire (NEG=0, POS=2, NEUTRAL=1 ignoré)
ARABIC_BINARY_DATASET = "arbml/Arabic_Sentiment_Twitter_Corpus"
# Darija (dialecte marocain) : utiliser dialectal Arabic ou créer synthétique
DARIJA_DATASET = "QCRI/BERT-Base-Multilingual-Cased-finetuned-Arabic"  # fallback

# Translitération Arabe → Arabizi (dialecte marocain: latino + chiffres)
ARABIC_TO_ARABIZI = {
    'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': '7', 'خ': 'kh',
    'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'ch', 'ص': 's9',
    'ض': 'd9', 'ط': 't9', 'ظ': 'z7', 'ع': '3', 'غ': 'gh', 'ف': 'f', 'ق': '9',
    'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
    'ة': 'a',
}


def stars_to_sentiment(star_label: int) -> int | None:
    """
    Labels Amazon MTEB : 0..4 pour 1..5 étoiles (ordre croissant).
    Binaire : 1-2 étoiles -> négatif (0), 4-5 -> positif (2).
    3 étoiles -> IGNORÉ (None) [classe neutral supprimée].
    """
    if star_label <= 1:
        return 0
    if star_label == 2:  # 3 étoiles = neutral, on ignore
        return None
    return 2


def load_amazon_lang(lang: str, max_rows: int | None) -> Dataset:
    url = f"{AMAZON_JSONL_BASE}/{lang}/train.jsonl"
    ds = load_dataset("json", data_files=url, split="train")
    # Le fichier brut est groupé par étoile : prendre les N premières lignes biaise le sentiment.
    ds = ds.shuffle(seed=42)
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds


def amazon_to_records(ds: Dataset, lang: str) -> list[dict]:
    rows = []
    for ex in ds:
        text = clean_text(ex["text"])
        if not is_valid_sample(text):
            continue
        y = stars_to_sentiment(int(ex["label"]))
        if y is None:  # Skip neutral
            continue
        rows.append({"text": text, "label": y, "lang": lang, "source": "amazon_mteb"})
    return rows


def _arabic_from_arbml(max_rows: int) -> list[dict]:
    """Charger arabe depuis arbml (binaire: neg/pos seulement)."""
    if max_rows <= 0:
        return []
    try:
        ds = load_dataset(ARABIC_BINARY_DATASET, split="train")
        ds = ds.shuffle(seed=43)
        ds = ds.select(range(min(max_rows, len(ds))))
        rows = []
        for ex in ds:
            text = clean_text(ex["tweet"])
            if not is_valid_sample(text):
                continue
            y = 0 if int(ex["label"]) == 0 else 2
            rows.append({"text": text, "label": y, "lang": "ar", "source": "arbml_twitter"})
        return rows
    except Exception as e:
        print(f"⚠️  Erreur chargement arabe arbml: {e}. Continuant sans...")
        return []


def _transliterate_arabic_to_arabizi(text: str) -> str:
    """Translitération Arabe → Arabizi (darija marocain: latino + chiffres)."""
    result = []
    for char in text:
        if char in ARABIC_TO_ARABIZI:
            result.append(ARABIC_TO_ARABIZI[char])
        else:
            result.append(char)
    return "".join(result).replace("  ", " ").strip()


def load_darija(max_rows: int | None) -> list[dict]:
    """
    Charger Darija en 2 écritures :
    1. Arabe (script arabe)
    2. Arabizi (translittération latino + chiffres marocaine)
    """
    rows = []
    try:
        ds = load_dataset(ARABIC_BINARY_DATASET, split="train")
        ds = ds.shuffle(seed=44)
        if max_rows:
            ds = ds.select(range(min(max_rows, len(ds))))
        
        for ex in ds:
            text = clean_text(ex["tweet"])
            if not is_valid_sample(text):
                continue
            y = 0 if int(ex["label"]) == 0 else 2
            
            # Version 1 : Darija en script Arabe
            rows.append({"text": text, "label": y, "lang": "darija", "source": "darija_arabic"})
            
            # Version 2 : Darija en Arabizi (latino + chiffres)
            arabizi_text = _transliterate_arabic_to_arabizi(text)
            if len(arabizi_text) > 5:  # Vérifier la qualité
                rows.append({"text": arabizi_text, "label": y, "lang": "darija_arabizi", "source": "darija_arabizi"})
        
        return rows
    except Exception as e:
        print(f"⚠️  Erreur chargement Darija: {e}. Utilisant fallback...")
        return []


def load_arabic(max_rows: int | None) -> list[dict]:
    """Charger uniquement arabe binaire (pas de neutral)."""
    return _arabic_from_arbml(max_rows or 8000)


def build_dataframe(
    max_per_lang: int | None = 5000,
    arabic_max: int | None = None,
    darija_max: int | None = None,
) -> pd.DataFrame:
    all_rows: list[dict] = []
    arabic_limit = arabic_max if arabic_max is not None else max_per_lang
    darija_limit = darija_max if darija_max is not None else max_per_lang

    # Charger les 3 langues Amazon (rapide, en parallèle possible)
    for lang in AMAZON_LANGS:
        print(f"⏳ Chargement {lang.upper()}...")
        ds = load_amazon_lang(lang, max_per_lang)
        all_rows.extend(amazon_to_records(ds, lang))

    # Charger Arabe (binaire, sans neutral)
    print("⏳ Chargement ARABE...")
    all_rows.extend(load_arabic(arabic_limit))
    
    # Charger Darija (2 écritures)
    print("⏳ Chargement DARIJA (arabe + latino)...")
    all_rows.extend(load_darija(darija_limit))
    
    # Créer dataframe et shuffle
    df = pd.DataFrame(all_rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Remapper les labels pour être continus (0, 2 → 0, 1)
    df["label"] = df["label"].replace({0: 0, 2: 1})
    
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Construire le dataset multilingue")
    p.add_argument("--max-per-lang", type=int, default=8000, help="Max lignes par langue (Amazon)")
    p.add_argument("--arabic-max", type=int, default=None, help="Max lignes arabe (défaut = max-per-lang)")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "processed"))
    p.add_argument("--csv", action="store_true", help="Exporter aussi un CSV")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(max_per_lang=args.max_per_lang, arabic_max=args.arabic_max)
    parquet_path = out_dir / "dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Écrit : {parquet_path} ({len(df)} lignes)")
    print(df.groupby(["lang", "label"]).size().unstack(fill_value=0))

    if args.csv:
        csv_path = out_dir / "dataset.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Écrit : {csv_path}")


if __name__ == "__main__":
    main()
