# Exécute tout le pipeline de A à Z

Write-Host "=========================================="
Write-Host "1. Construction du Dataset Multilingue..."
Write-Host "=========================================="
python scripts\build_dataset.py

Write-Host "=========================================="
Write-Host "2. Validation du Prétraitement..."
Write-Host "=========================================="
python scripts\run_preprocess.py

Write-Host "=========================================="
Write-Host "3. Exploration des Données (EDA)..."
Write-Host "=========================================="
python scripts\run_eda.py

Write-Host "=========================================="
Write-Host "4. Entraînement / Fine-tuning..."
Write-Host "=========================================="
# Si vous voulez l'entraîner sur un échantillon très réduit pour tester que le code marche, ajoutez l'argument --debug :
# python -m src.train --debug
# Pour entraîner rapidement avec 8-bit optimizer :
# python -m src.train --use-8bit
python -m src.train --use-8bit

Write-Host "=========================================="
Write-Host "5. Évaluation du Modèle..."
Write-Host "=========================================="
# Génère : classification report, confusion matrix, ROC/AUC, visualisations
# Pour un test rapide : python -m src.evaluate --debug
python -m src.evaluate

Write-Host "=========================================="
Write-Host "6. Test de l'Inférence (Démonstration)..."
Write-Host "=========================================="
python -m src.inference --demo

Write-Host "=========================================="
Write-Host "🎉 Pipeline Terminé avec Succès!"
Write-Host "=========================================="
