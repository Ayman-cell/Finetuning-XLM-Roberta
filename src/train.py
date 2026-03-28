"""Script d'entraînement (fine-tuning) du modèle de sentiment multilingue."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Configure PyTorch memory for fragmentation avoidance on 6GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ajouter src au PYTHONPATH si ce n'est pas le cas pour les imports relatifs
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.config import ID2LABEL, LABEL2ID, MAX_LENGTH, MODEL_DIR, MODEL_NAME, NUM_LABELS, RAW_DATA_PATH


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2, help="Nombre d'epochs (réduit pour rapidité)")
    p.add_argument("--batch-size", type=int, default=2, help="Taille de batch par device (réduit pour GPUs 6GB)")
    p.add_argument("--grad-accum", type=int, default=8, help="Étapes d'accumulation de gradients")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--no-fp16", action="store_true", help="Désactiver la précision mixte (FP16)")
    p.add_argument("--use-8bit", action="store_true", help="Utiliser AdamW 8-bit pour économiser la mémoire")
    p.add_argument("--debug", action="store_true", help="Run with a small subset for debugging")
    args = p.parse_args()

    # Load data
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Les données sont introuvables: {RAW_DATA_PATH}. Lancez d'abord: python scripts/build_dataset.py")

    df = pd.read_parquet(RAW_DATA_PATH)
    if args.debug:
        print("DEBUG MODE: Using only 200 samples")
        df = df.sample(min(200, len(df)), random_state=42)

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(df)
    
    # Train/Val/Test Split (80-10-10)
    train_testval = hf_dataset.train_test_split(test_size=0.2, seed=42)
    testval_split = train_testval["test"].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = {
        "train": train_testval["train"],
        "validation": testval_split["train"],
        "test": testval_split["test"],
    }

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=MAX_LENGTH
        )

    tokenized_datasets = {}
    for split, ds in dataset_dict.items():
        tokenized_datasets[split] = ds.map(tokenize_function, batched=True, remove_columns=["text", "lang", "source"])

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS, 
        id2label=ID2LABEL, 
        label2id=LABEL2ID
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Optimizer choice for memory efficiency
    optim = "adamw_8bit" if args.use_8bit else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR.parent / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),  # Reduce eval batch size even more
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1 if args.debug else args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="none",
        fp16=torch.cuda.is_available() and not args.no_fp16,
        gradient_checkpointing=True,
        optim=optim,
        max_grad_norm=1.0,
        logging_steps=50,
        save_total_limit=2,  # Keep only 2 latest checkpoints to save space
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Début de l'entraînement...")
    trainer.train()

    # Clear GPU cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Sauvegarde du modèle dans {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Clear cache again before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Évaluation sur le jeu de test...")
    metrics = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
