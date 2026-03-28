# Multilingual Sentiment Analysis with XLM-RoBERTa + Darija Support

A production-ready fine-tuned transformer model for **5-language sentiment analysis** (English, French, Spanish, Arabic) + **Darija dialect** in both **Arabic script** and **Arabizi** (Latin transliteration).

## 🎯 Features

- **5 Languages + Darija (2 variants)**:
  - English, French, Spanish (Amazon reviews)
  - Arabic (binary: negative/positive)
  - Darija (Moroccan dialect) in Arabic script + Arabizi

- **Optimized for 6GB GPUs**:
  - Batch size = 2, gradient accumulation = 8
  - 8-bit AdamW optimizer support
  - Memory-efficient training with `PYTORCH_CUDA_ALLOC_CONF`

- **Binary Classification** (Negative/Positive):
  - Neutral class removed for better dataset balance
  - 2 labels: `negative` (0), `positive` (1)

- **Complete ML Pipeline**:
  1. Dataset construction from Hugging Face
  2. Preprocessing & multilingual validation
  3. EDA with visualizations
  4. Fine-tuning with Transformers
  5. Evaluation with metrics & confusion matrix
  6. Inference demo

## 📊 Dataset

**Total samples**: ~40,000+ (balanced across languages)

| Language | Samples | Source |
|----------|---------|--------|
| English | 8,000 | Amazon MTEB |
| Français | 8,000 | Amazon MTEB |
| Español | 8,000 | Amazon MTEB |
| العربية | 8,000 | ArBML Twitter Corpus |
| Darija (Arabic) | 3,000+ | ArBML (dialectal) |
| Darija (Arabizi) | 3,000+ | Transliterated |

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full Pipeline

```powershell
.\run_pipeline.ps1
```

Individual steps:

```bash
# 1. Build dataset (EN + FR + ES + AR + Darija)
python scripts/build_dataset.py

# 2. Preprocess & validate multilingually
python scripts/run_preprocess.py

# 3. EDA & visualizations
python scripts/run_eda.py

# 4. Train (optimized for 6GB GPU)
python -m src.train --use-8bit --epochs 2

# 5. Evaluate
python -m src.evaluate

# 6. Demo inference
python -m src.inference --demo
```

## ⬇️ Pre-trained Models

### 🤗 Download from Hugging Face Hub

The trained multilingual sentiment model is **now available** on Hugging Face:

**📍 Model:** [`ChillyKuw/xlm-roberta-multilingual-sentiment`](https://huggingface.co/ChillyKuw/xlm-roberta-multilingual-sentiment)

#### Quick Start - Load Pre-trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_id = "ChillyKuw/xlm-roberta-multilingual-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Prediction
texts = [
    "This product is amazing!",           # English
    "Ce produit est incroyable!",          # French
    "¡Este producto es increíble!",        # Spanish
    "هذا المنتج رائع جداً",                 # Arabic
    "المنتج زين بزاف",                     # Darija (Arabic script)
    "l bnt3 jmila bezzaf"                  # Darija (Arabizi)
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

labels = {0: "negative", 1: "positive"}
for text, pred in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {labels[pred.item()]}\n")
```

#### Download Files (CLI)

```bash
# Download model files
huggingface-cli download ChillyKuw/xlm-roberta-multilingual-sentiment --local-dir ./models/sentiment_model

# Or using Python
from huggingface_hub import snapshot_download
snapshot_download("ChillyKuw/xlm-roberta-multilingual-sentiment", local_dir="./models/sentiment_model")
```

---

## 📊 Model Performance

**Overall Metrics:**
- **Accuracy**: 94.01%
- **Precision (Macro)**: 94.01%
- **Recall (Macro)**: 94.01%
- **F1-Score**: 94.01%
- **AUC-ROC**: 0.9839

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 94.19% | 93.85% | 94.02% | 2,161 |
| Positive | 93.82% | 94.17% | 93.99% | 2,145 |

**Per-Language Accuracy:**

| Language | Accuracy | Samples |
|----------|----------|---------|
| English (en) | 93.0% | 819 |
| French (fr) | 95.5% | 829 |
| Spanish (es) | 92.0% | 767 |
| Arabic (ar) | 93.8% | 645 |
| Darija (Arabic script) | 94.4% | 625 |
| Darija (Arabizi) | 95.5% | 621 |

### 📈 Visualizations

#### Confusion Matrix
- **Balanced predictions** across classes
- True Positive Rate: 94.17% (positive)
- True Negative Rate: 93.85% (negative)
- False Positive Rate: 6.15% (minimal misclassifications)

#### ROC/AUC Curve
- **Area Under Curve: 0.9839** (excellent discrimination)
- Model clearly separates positive and negative sentiments

#### Confidence Distribution
- **Mean confidence: 97.9%** - model is very confident
- High-confidence predictions lead to reliable classifications

---

## 🧪 Test Cases & Examples

### Example 1: English Review
```python
text = "This movie was absolutely fantastic! Best film I've seen all year."
# Expected: positive ✅
# Confidence: 99.2%
```

### Example 2: Multilingual Sentiment
```python
examples = {
    "English": "The product broke after two days. Very disappointed.",
    "French": "Le service client est excellent et rapide.",
    "Spanish": "El precio es demasiado alto para la calidad.",
    "Arabic": "الخدمة سيئة جداً والموظفين غير مهنيين",
    "Darija": "الطاجين حضّير بزاف بصح الديليفري طول",
}

# All correctly predicted ✅
```

### Example 3: Mixed Language (Code-switching)
```python
text = "Good quality but shipping was very slow 😞"
# Prediction: negative (due to "slow" and negative emoji)
# Confidence: 96.8%
```

### Example 4: Darija Variants (Same meaning, different scripts)
```python
darija_arabic = "هاد المنتج حسن بزاف"      # Arabic script
darija_arabizi = "had lmntaj hsan bzaf"   # Arabizi (Latin)

# Both predicted: positive ✅
# Shows excellent code-switching capability
```

---

### Upload Your Own Models (Optional)

If you want to upload your trained models to Hugging Face:

```bash
# Create a new model on HF and get token from https://huggingface.co/settings/tokens
python upload_to_huggingface.py --token "your_token_here" --username "your_username"
```

---

Edit [src/config.py](src/config.py):

```python
MODEL_NAME = "FacebookAI/xlm-roberta-base"  # Multilingual XLMR
MAX_LENGTH = 96  # Optimized for GPU memory
NUM_LABELS = 2  # Negative / Positive
SUPPORTED_LANGS = ["en", "fr", "es", "ar", "darija", "darija_arabizi"]
```

## 📈 Training Options

```bash
# Quick training (GPU optimized)
python -m src.train --use-8bit --epochs 2

# Debug mode (small dataset)
python -m src.train --debug

# Custom parameters
python -m src.train \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --use-8bit
```

## 🧹 Preprocessing Pipeline

The [src/preprocess.py](src/preprocess.py) module:

- Removes URLs, mentions, control characters
- Handles multilingual (RTL) scripts correctly
- Validates text length (5-5000 chars)
- Unicode normalization (NFC)

```python
from src.preprocess import clean_text, is_valid_sample

text = "Check https://example.com @user Great product! 🎉"
cleaned = clean_text(text)  # "Check Great product!"
is_valid = is_valid_sample(cleaned)  # True
```

## 📁 Project Structure

```
sentiment_multilingual/
├── data/
│   └── processed/
│       └── dataset.parquet          # Preprocessed dataset
├── models/
│   ├── sentiment_model/             # Fine-tuned model
│   └── checkpoints/                 # Training checkpoints
├── outputs/
│   ├── figures/                     # EDA visualizations
│   └── metrics/                     # Evaluation metrics
├── src/
│   ├── config.py                    # Configuration
│   ├── preprocess.py                # Text preprocessing
│   ├── train.py                     # Fine-tuning script
│   ├── evaluate.py                  # Evaluation
│   └── inference.py                 # Inference demo
├── scripts/
│   ├── build_dataset.py             # Download & build dataset
│   ├── run_preprocess.py            # Preprocess & save
│   └── run_eda.py                   # Analysis & plots
├── requirements.txt
└── run_pipeline.ps1                 # Full pipeline automation
```

## 🔍 Example Inference

```python
python -m src.inference --demo
```

**Output**:
```
Text: "Ce produit est incroyable!"
Language: fr
Prediction: positive (98.2% confidence)
---
Text: "المنتج سيء جداً"
Language: ar
Prediction: negative (95.1% confidence)
---
Text: "المنتج زين بزاف" (Darija)
Language: darija
Prediction: positive (92.7% confidence)
```

## 📊 Evaluation

```bash
python -m src.evaluate
```

Generates:
- Classification report (precision/recall/F1)
- Confusion matrix visualization
- ROC/AUC scores
- Per-language metrics
- Plots saved to `outputs/figures/`

## 🎓 Model Details

- **Base Model**: `FacebookAI/xlm-roberta-base` (12 layers, 110M params)
- **Task**: Binary sequence classification
- **Optimization**: AdamW (8-bit), gradient checkpointing
- **Precision**: FP16 mixed precision
- **Loss**: Cross-entropy

## 💾 Outputs

After running the pipeline:

```
outputs/
├── figures/
│   ├── count_by_lang_label.png       # Distribution by language
│   ├── length_boxplot.png            # Text length analysis
│   └── sentiment_distribution.png    # Overall sentiment dist.
└── metrics/
    ├── evaluation_metrics.json       # Main metrics
    ├── per_language_metrics.json     # Per-language breakdown
    └── auc_scores.json               # ROC/AUC scores

models/
├── sentiment_model/                  # Final trained model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── special_tokens_map.json
└── checkpoints/                      # Training checkpoints
    ├── checkpoint-100/
    ├── checkpoint-500/
    └── ...
```

## 🛠️ GPU Memory Optimization

**For 6GB GPUs**:

1. ✅ **Already applied**:
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - Batch size = 2
   - Gradient accumulation = 8
   - FP16 mixed precision
   - Gradient checkpointing
   - 8-bit AdamW optimizer
   - Reduced MAX_LENGTH (96 vs 512)

2. **If still OOM**:
   ```bash
   python -m src.train --batch-size 1 --grad-accum 16 --use-8bit
   ```

## 📦 Dependencies

- torch >= 2.0.0
- transformers >= 4.36.0
- datasets >= 2.16.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- seaborn >= 0.13.0

See [requirements.txt](requirements.txt) for full list.

## 🤝 Contributing

This project demonstrates:
- Multilingual NLP best practices
- Memory-efficient training on limited GPUs
- Pipeline automation (build → preprocess → train → evaluate)
- Support for dialectal Arabic (Darija)

## 📝 License

MIT License

## 🙏 Acknowledgments

- **Datasets**: Hugging Face (Amazon MTEB, ArBML, ASTD)
- **Model**: FacebookAI XLM-RoBERTa
- **Framework**: Hugging Face Transformers

---

**Questions?** Check the `.gitignore` and logs in `outputs/` for debugging.
