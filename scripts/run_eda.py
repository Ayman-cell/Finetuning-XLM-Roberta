#!/usr/bin/env python3
"""EDA : statistiques, longueurs, distribution par langue et sentiment, graphiques."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_DATA_PATH, LABEL_NAMES, ID2LABEL, FIGURES_DIR

FIG = FIGURES_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(RAW_DATA_PATH))
    args = p.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}\nLancez d'abord:\n  1. python scripts/build_dataset.py\n  2. python scripts/run_preprocess.py")

    print(f"📖 Chargement dataset: {path}")
    df = pd.read_parquet(path)
    print(f"✓ {len(df)} samples chargés\n")
    
    FIG.mkdir(parents=True, exist_ok=True)

    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()

    print("=" * 70)
    print("📊 APERÇU DU DATASET")
    print("=" * 70)
    print(df.head(3).to_string())
    
    print("\n" + "=" * 70)
    print("📈 DISTRIBUTION PAR LANGUE ET SENTIMENT")
    print("=" * 70)
    ct = pd.crosstab(df["lang"], df["label"], margins=True)
    # Renommer les colonnes avec les labels
    ct_renamed = ct.copy()
    for col in ct_renamed.columns:
        if col in ID2LABEL:
            ct_renamed = ct_renamed.rename(columns={col: ID2LABEL[col]})
    print(ct_renamed)

    print("\n" + "=" * 70)
    print("📏 LONGUEUR (caractères) PAR LANGUE")
    print("=" * 70)
    print(df.groupby("lang")["n_chars"].describe())

    # Graphiques
    print("\n" + "=" * 70)
    print("🎨 GÉNÉRATION DES GRAPHIQUES")
    print("=" * 70)
    
    # Graphique 1: Distribution par langue et label
    plt.figure(figsize=(10, 5))
    unique_labels = sorted(df["label"].unique())
    label_names = [ID2LABEL.get(l, str(l)) for l in unique_labels]
    sns.countplot(data=df, x="lang", hue="label", hue_order=unique_labels, palette="Set2")
    plt.title("Distribution des sentiments par langue", fontsize=12, fontweight="bold")
    plt.legend(title="Sentiment", labels=label_names)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG / "count_by_lang_label.png", dpi=150)
    print("✓ count_by_lang_label.png")
    plt.close()

    # Graphique 2: Longueur des textes
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="lang", y="n_chars", palette="Set2")
    plt.title("Longueur des textes (caractères) par langue", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG / "length_boxplot.png", dpi=150)
    print("✓ length_boxplot.png")
    plt.close()

    # Graphique 3: Distribution globale des sentiments
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="label", palette="Set2", order=unique_labels)
    plt.title("Distribution globale des sentiments", fontsize=12, fontweight="bold")
    plt.xlabel("Sentiment")
    ax = plt.gca()
    ax.set_xticklabels(label_names)
    plt.tight_layout()
    plt.savefig(FIG / "sentiment_distribution.png", dpi=150)
    print("✓ sentiment_distribution.png")
    plt.close()

    print(f"\n📁 Graphiques sauvegardés: {FIG}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
