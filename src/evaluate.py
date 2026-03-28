"""Évaluation complète du modèle de sentiment multilingue.

Génère :
  - Classification report (accuracy, precision, recall, F1)
  - Matrice de confusion
  - Courbes ROC / AUC (one-vs-rest pour multi-classe)
  - Visualisation des prédictions sur le jeu de test
  - Distribution des probabilités de confiance
  - Métriques par langue
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")  # Backend non-interactif – compatible serveurs & scripts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import FIGURES_DIR, ID2LABEL, LABEL2ID, LABEL_NAMES, MAX_LENGTH, METRICS_DIR, MODEL_DIR, NUM_LABELS, RAW_DATA_PATH

# ──────────────────────────────────────
# Style global pour les figures
# ──────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#475569",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "font.family": "sans-serif",
    "font.size": 11,
})

PALETTE = ["#f43f5e", "#38bdf8"]  # negative, positive
PALETTE_CMAP = sns.color_palette(PALETTE, as_cmap=False)


# ──────────────────────────────────────
# Helpers
# ──────────────────────────────────────
def _save_fig(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔ Figure sauvegardée : {path}")
    return path


def _get_predictions(model, tokenizer, dataset, device, batch_size: int = 32):
    """Retourne (y_true, y_pred, y_probs) sur le dataset."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        enc = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = batch["label"]

        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        all_labels.extend(labels)
        all_preds.extend(preds.tolist())
        all_probs.append(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


# ──────────────────────────────────────
# 1. Classification Report
# ──────────────────────────────────────
def print_and_save_classification_report(y_true, y_pred):
    report_str = classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)

    print("\n" + "=" * 60)
    print("              CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_str)

    # Résumé compact
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro")
    rec_macro = recall_score(y_true, y_pred, average="macro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    summary = {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": report_dict,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / "evaluation_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✔ Métriques sauvegardées : {out}")

    return summary


# ──────────────────────────────────────
# 2. Confusion Matrix
# ──────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Matrice de Confusion", fontsize=16, fontweight="bold", color="#f8fafc")

    # Valeurs absolues
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="rocket_r",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[0], linewidths=0.5, linecolor="#475569",
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Valeurs Absolues", fontsize=13, pad=10)
    axes[0].set_xlabel("Prédiction")
    axes[0].set_ylabel("Vérité")

    # Normalisée
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="rocket_r",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[1], linewidths=0.5, linecolor="#475569",
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Normalisée (par ligne)", fontsize=13, pad=10)
    axes[1].set_xlabel("Prédiction")
    axes[1].set_ylabel("Vérité")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "confusion_matrix")


# ──────────────────────────────────────
# 3. ROC / AUC Curves
# ──────────────────────────────────────
def plot_roc_auc(y_true, y_probs):
    n_classes = len(LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title("Courbe ROC", fontsize=15, fontweight="bold", color="#f8fafc", pad=12)

    colors = PALETTE
    auc_scores = {}

    if n_classes == 2:
        # Classification binaire : ROC sur la classe positive (index 1)
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        auc_scores["overall"] = round(roc_auc, 4)
        ax.plot(fpr, tpr, color="#38bdf8", lw=2.5, label=f"AUC = {roc_auc:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.15, color="#38bdf8")
    else:
        # Multi-classe : one-vs-rest
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i, (label, color) in enumerate(zip(LABEL_NAMES, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[label] = round(roc_auc, 4)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{label}  (AUC = {roc_auc:.4f})")

        # Micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_probs.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, color="#fbbf24", lw=2, linestyle="--",
                label=f"micro-avg  (AUC = {auc_micro:.4f})")

    ax.plot([0, 1], [0, 1], "w--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.legend(loc="lower right", fontsize=10, facecolor="#1e293b", edgecolor="#475569")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(True, alpha=0.3)

    _save_fig(fig, "roc_auc_curves")

    # AUC score
    if n_classes == 2:
        auc_scores["binary_auc"] = auc_scores["overall"]
    else:
        try:
            macro_auc = roc_auc_score(y_bin, y_probs, multi_class="ovr", average="macro")
        except Exception:
            macro_auc = np.mean(list(auc_scores.values()))
        auc_scores["macro_avg"] = round(macro_auc, 4)

    out = METRICS_DIR / "auc_scores.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(auc_scores, f, indent=2)
    print(f"  ✔ AUC scores sauvegardés : {out}")


# ──────────────────────────────────────
# 4. Confidence Distribution
# ──────────────────────────────────────
def plot_confidence_distribution(y_true, y_pred, y_probs):
    max_probs = y_probs.max(axis=1)
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribution de la Confiance du Modèle", fontsize=16, fontweight="bold", color="#f8fafc")

    # Histogramme global
    axes[0].hist(max_probs, bins=40, color="#8b5cf6", edgecolor="#0f172a", alpha=0.85)
    axes[0].axvline(max_probs.mean(), color="#fbbf24", ls="--", lw=2, label=f"Moyenne: {max_probs.mean():.3f}")
    axes[0].set_title("Distribution Globale", fontsize=13, pad=10)
    axes[0].set_xlabel("Probabilité Max (confiance)")
    axes[0].set_ylabel("Nombre d'échantillons")
    axes[0].legend(facecolor="#1e293b", edgecolor="#475569")

    # Correct vs Incorrect
    axes[1].hist(max_probs[correct], bins=40, color="#34d399", alpha=0.7, label="Correct", edgecolor="#0f172a")
    axes[1].hist(max_probs[~correct], bins=40, color="#f43f5e", alpha=0.7, label="Incorrect", edgecolor="#0f172a")
    axes[1].set_title("Correct vs Incorrect", fontsize=13, pad=10)
    axes[1].set_xlabel("Probabilité Max (confiance)")
    axes[1].set_ylabel("Nombre d'échantillons")
    axes[1].legend(facecolor="#1e293b", edgecolor="#475569")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "confidence_distribution")


# ──────────────────────────────────────
# 5. Test Set Visualisation
# ──────────────────────────────────────
def plot_test_predictions_overview(y_true, y_pred, y_probs, langs):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Visualisation des Prédictions sur le Jeu de Test",
                 fontsize=18, fontweight="bold", color="#f8fafc", y=0.98)

    # ── (0,0) Distribution des prédictions vs vérité ──
    ax = axes[0, 0]
    x = np.arange(len(LABEL_NAMES))
    width = 0.35
    true_counts = np.bincount(y_true, minlength=len(LABEL_NAMES))
    pred_counts = np.bincount(y_pred, minlength=len(LABEL_NAMES))
    bars_true = ax.bar(x - width / 2, true_counts, width, label="Vérité", color="#38bdf8", edgecolor="#0f172a")
    bars_pred = ax.bar(x + width / 2, pred_counts, width, label="Prédiction", color="#a78bfa", edgecolor="#0f172a")
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_title("Distribution : Vérité vs Prédictions", fontsize=13, pad=10)
    ax.legend(facecolor="#1e293b", edgecolor="#475569")
    # Ajouter les annotations
    for bar in bars_true:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9, color="#e2e8f0")
    for bar in bars_pred:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9, color="#e2e8f0")

    # ── (0,1) Per-class F1 scores ──
    ax = axes[0, 1]
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    f1_scores = [report[name]["f1-score"] for name in LABEL_NAMES]
    bars = ax.barh(LABEL_NAMES, f1_scores, color=PALETTE, edgecolor="#0f172a", height=0.5)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=11, fontweight="bold", color="#fbbf24")
    ax.set_xlim(0, 1.15)
    ax.set_title("F1-Score par Classe", fontsize=13, pad=10)
    ax.axvline(1.0, color="#475569", ls=":", alpha=0.5)

    # ── (1,0) Accuracy per language ──
    ax = axes[1, 0]
    if langs is not None and len(langs) == len(y_true):
        unique_langs = sorted(set(langs))
        lang_acc = []
        lang_counts = []
        for lang in unique_langs:
            mask = np.array([l == lang for l in langs])
            if mask.sum() > 0:
                lang_acc.append(accuracy_score(y_true[mask], y_pred[mask]))
                lang_counts.append(int(mask.sum()))
            else:
                lang_acc.append(0)
                lang_counts.append(0)

        lang_colors = sns.color_palette("viridis", len(unique_langs))
        bars = ax.bar(unique_langs, lang_acc, color=lang_colors, edgecolor="#0f172a")
        for bar, acc, cnt in zip(bars, lang_acc, lang_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.3f}\n(n={cnt})", ha="center", va="bottom", fontsize=9, color="#e2e8f0")
        ax.set_ylim(0, 1.15)
        ax.set_title("Accuracy par Langue", fontsize=13, pad=10)
        ax.set_ylabel("Accuracy")
    else:
        ax.text(0.5, 0.5, "Colonne 'lang' non disponible", ha="center", va="center",
                fontsize=13, color="#94a3b8", transform=ax.transAxes)
        ax.set_title("Accuracy par Langue", fontsize=13, pad=10)

    # ── (1,1) Probabilités moyennes par classe prédite ──
    ax = axes[1, 1]
    mean_probs_per_class = []
    for c in range(len(LABEL_NAMES)):
        mask = y_pred == c
        if mask.sum() > 0:
            mean_probs_per_class.append(y_probs[mask].mean(axis=0))
        else:
            mean_probs_per_class.append(np.zeros(len(LABEL_NAMES)))
    mean_probs_per_class = np.array(mean_probs_per_class)

    sns.heatmap(
        mean_probs_per_class, annot=True, fmt=".3f",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        cmap="magma", ax=ax, linewidths=0.5, linecolor="#475569",
        vmin=0, vmax=1, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Prob. Moyenne par Classe Prédite", fontsize=13, pad=10)
    ax.set_xlabel("Classe (probabilité)")
    ax.set_ylabel("Classe prédite")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "test_predictions_overview")


# ──────────────────────────────────────
# 6. Per-language detailed metrics
# ──────────────────────────────────────
def print_per_language_metrics(y_true, y_pred, langs):
    if langs is None or len(langs) != len(y_true):
        print("  ⚠ Colonne 'lang' non disponible, métriques par langue ignorées.")
        return

    unique_langs = sorted(set(langs))
    print("\n" + "=" * 60)
    print("          MÉTRIQUES PAR LANGUE")
    print("=" * 60)

    rows = []
    for lang in unique_langs:
        mask = np.array([l == lang for l in langs])
        yt, yp = y_true[mask], y_pred[mask]
        row = {
            "lang": lang,
            "n": int(mask.sum()),
            "accuracy": round(accuracy_score(yt, yp), 4),
            "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
            "precision_macro": round(precision_score(yt, yp, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(yt, yp, average="macro", zero_division=0), 4),
        }
        rows.append(row)
        print(f"  [{lang}]  n={row['n']:>5}  acc={row['accuracy']:.4f}  "
              f"F1={row['f1_macro']:.4f}  P={row['precision_macro']:.4f}  R={row['recall_macro']:.4f}")

    out = METRICS_DIR / "per_language_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"  ✔ Métriques par langue sauvegardées : {out}")


# ──────────────────────────────────────
# Main
# ──────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="Évaluation complète du modèle de sentiment.")
    p.add_argument("--model-dir", type=str, default=str(MODEL_DIR), help="Répertoire du modèle fine-tuné")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--debug", action="store_true", help="Évaluer sur un petit sous-ensemble (200 samples)")
    args = p.parse_args()

    # ── Charger les données ──
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Données introuvables : {RAW_DATA_PATH}")

    df = pd.read_parquet(RAW_DATA_PATH)
    if args.debug:
        print("⚡ DEBUG MODE: 200 échantillons uniquement")
        df = df.sample(min(200, len(df)), random_state=42)

    # Reproduire le même split que train.py (seed=42)
    hf_dataset = Dataset.from_pandas(df)
    train_testval = hf_dataset.train_test_split(test_size=0.2, seed=42)
    testval_split = train_testval["test"].train_test_split(test_size=0.5, seed=42)
    test_dataset = testval_split["test"]

    print(f"📊 Taille du jeu de test : {len(test_dataset)}")

    # ── Charger le modèle ──
    model_path = Path(args.model_dir)
    print(f"📦 Chargement du modèle depuis : {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"🖥  Device : {device}")

    # ── Récupérer les prédictions ──
    # On reconstruit un dataset avec la colonne text pour la prédiction
    test_df = test_dataset.to_pandas()
    langs = test_df["lang"].tolist() if "lang" in test_df.columns else None

    print("\n🔄 Calcul des prédictions sur le jeu de test...")
    y_true, y_pred, y_probs = _get_predictions(model, tokenizer, test_dataset, device, args.batch_size)

    # ── Rapport ──
    print_and_save_classification_report(y_true, y_pred)

    # ── Visualisations ──
    print("\n🎨 Génération des visualisations...")
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_auc(y_true, y_probs)
    plot_confidence_distribution(y_true, y_pred, y_probs)
    plot_test_predictions_overview(y_true, y_pred, y_probs, langs)

    # ── Métriques par langue ──
    print_per_language_metrics(y_true, y_pred, langs)

    print("\n" + "=" * 60)
    print("✅ Évaluation terminée ! Résultats dans outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
