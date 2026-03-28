#!/usr/bin/env python3
"""
Upload trained models to Hugging Face Model Hub
Requirements: pip install huggingface_hub transformers
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def upload_model_to_hf(local_model_path, hf_model_id, hf_token):
    """
    Upload a model to Hugging Face Hub
    
    Args:
        local_model_path: Local path to model directory
        hf_model_id: Hugging Face model ID (e.g., "username/model-name")
        hf_token: Hugging Face API token
    """
    print(f"\n📤 Uploading {local_model_path} to {hf_model_id}...")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    
    # Push to Hub
    model.push_to_hub(hf_model_id, token=hf_token, private=False)
    tokenizer.push_to_hub(hf_model_id, token=hf_token, private=False)
    
    print(f"✅ Successfully uploaded to https://huggingface.co/{hf_model_id}")
    return hf_model_id

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face")
    parser.add_argument("--token", required=True, help="Hugging Face API token (get from https://huggingface.co/settings/tokens)")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models" / "sentiment_model"
    
    if not models_dir.exists():
        print(f"❌ Model not found at {models_dir}")
        exit(1)
    
    # Upload final sentiment model
    hf_model_id = f"{args.username}/xlm-roberta-multilingual-sentiment"
    
    try:
        uploaded_id = upload_model_to_hf(
            str(models_dir),
            hf_model_id,
            args.token
        )
        print(f"\n🎉 Model uploaded successfully!")
        print(f"📍 Model URL: https://huggingface.co/{uploaded_id}")
        print(f"💻 Usage: model = AutoModel.from_pretrained('{uploaded_id}')")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        exit(1)
