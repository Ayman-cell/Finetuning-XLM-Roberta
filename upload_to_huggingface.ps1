# Upload models to Hugging Face
# Usage: .\upload_to_huggingface.ps1 -HFToken "your_token" -HFUsername "your_username"

param(
    [Parameter(Mandatory=$true)]
    [string]$HFToken,
    
    [Parameter(Mandatory=$true)]
    [string]$HFUsername
)

Write-Host "=========================================="
Write-Host "📤 Uploading Models to Hugging Face..."
Write-Host "=========================================="

# Check if Python is available
python --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found. Please install Python."
    exit 1
}

# Install required packages
Write-Host "📦 Installing required packages..."
pip install huggingface_hub transformers safetensors -q

# Run upload script
Write-Host ""
python upload_to_huggingface.py --token $HFToken --username $HFUsername

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Upload completed successfully!"
    Write-Host "📍 Check your repository: https://huggingface.co/$HFUsername"
} else {
    Write-Host "❌ Upload failed"
    exit 1
}
