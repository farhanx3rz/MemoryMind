$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Installing RECALL ML dependencies..."
python -m pip install -r requirements.txt

Write-Host "Training DistilBERT topic classifier..."
python -m ml.train

Write-Host "Evaluating trained classifier..."
python -m ml.evaluate

Write-Host "Training and evaluation complete."
Write-Host "Model: ml/models/recall-topic-classifier/"
