"""Configuration pour analyses de sentiment multilingue + Darija (2 écritures)."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "processed" / "dataset.parquet"
MODEL_DIR = ROOT_DIR / "models" / "sentiment_model"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"

# Model configuration
MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 96  # Optimisé pour GPU 6GB

# Labels (2 classes : négatif / positif, NEUTRAL SUPPRIMÉ)
NUM_LABELS = 2
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}
LABEL_NAMES = list(LABEL2ID.keys())

# Langues supportées : En, Fr, Es, Ar + Darija (arabe + arabizi)
SUPPORTED_LANGS = ["en", "fr", "es", "ar", "darija", "darija_arabizi"]
