#!/usr/bin/env python3
"""
Script de prétraitement du dataset.
- Applique clean_text à tous les samples
- Filtre les samples invalides (trop court, trop long, vide)
- Sauvegarde le dataset prétraité
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Ajouter src au PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_text, is_valid_sample
from src.config import RAW_DATA_PATH


def preprocess_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Charger le dataset brut, appliquer le prétraitement, et sauvegarder.
    """
    print(f"📖 Chargement du dataset brut: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset brut introuvable: {input_path}")
    
    df = pd.read_parquet(input_path)
    print(f"   → {len(df)} samples chargés")
    
    # Statistiques avant
    print(f"\n📊 AVANT prétraitement:")
    print(f"   - Langues: {df['lang'].unique()}")
    print(f"   - Distribution: \n{df.groupby(['lang', 'label']).size().unstack(fill_value=0)}")
    
    # Appliquer le nettoyage
    print(f"\n🧹 Nettoyage du texte...")
    df["text"] = df["text"].apply(clean_text)
    
    # Filtrer les samples invalides
    print(f"⚠️  Filtrage des samples invalides...")
    before = len(df)
    df = df[df["text"].apply(is_valid_sample)].reset_index(drop=True)
    filtered_out = before - len(df)
    print(f"   → {filtered_out} samples rejetés")
    
    if len(df) == 0:
        raise ValueError("Aucun sample valide après prétraitement!")
    
    # Statistiques après
    print(f"\n📊 APRÈS prétraitement:")
    print(f"   - {len(df)} samples valides")
    print(f"   - Distribution: \n{df.groupby(['lang', 'label']).size().unstack(fill_value=0)}")
    
    # Sauvegarder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)
    print(f"\n✅ Dataset prétraité sauvegardé: {output_path}")
    
    return df


def main() -> None:
    """Valider et prétraiter le dataset."""
    
    print("=" * 70)
    print("🧹 PRÉTRAITEMENT DU DATASET MULTILINGUE")
    print("=" * 70 + "\n")
    
    # Test sur petits exemples
    test_cases = {
        "English": "Check out https://example.com @user This product is great!",
        "Français": "Regardez ce lien: www.example.fr @mention C'est incroyable!",
        "Español": "Mira esto https://ejemplo.es @usuario ¡Excelente producto!",
        "العربية": "تحقق من الرابط https://example.ar @مستخدم المنتج رائع جداً",
        "Darija": "شوف الحاج @ريحان المنتج زين بزاف",
    }
    
    print("📌 TESTS MULTILINGUES:")
    for lang, text in test_cases.items():
        cleaned = clean_text(text)
        is_valid = is_valid_sample(cleaned)
        print(f"   {lang:15} → Valide: {is_valid}")
    
    print("\n" + "=" * 70)
    print("📦 PRÉTRAITEMENT DU DATASET COMPLET")
    print("=" * 70 + "\n")
    
    try:
        df_processed = preprocess_dataset(
            input_path=RAW_DATA_PATH,
            output_path=RAW_DATA_PATH
        )
        print("\n✅ Prétraitement réussi!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERREUR: {e}")
        print("\n💡 Conseil: Lancez d'abord `python scripts/build_dataset.py`")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        sys.exit(1)
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
