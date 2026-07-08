$ErrorActionPreference = "Stop"

Write-Host "Starting preprocessing pipeline"
Write-Host "1. Preprocessing PrimeVul dataset"
python -u preprocess.py

Write-Host "2. Compiling processed datasets to LLVM"
python -u compiler.py

Write-Host "Preprocessing complete"
