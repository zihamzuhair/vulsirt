$ErrorActionPreference = "Stop"

Write-Host "1. Preprocessing test dataset"
python preprocess.py

Write-Host "2. Training B4 for one tiny smoke-test epoch"
python train.py --config test/test_config.yaml --baseline b4

Write-Host "3. Evaluating B4"
python evaluate.py --config test/test_config.yaml --baseline b4

Write-Host "4. Scanning test/sample.c"
python scanner.py --config test/test_config.yaml --baseline b4 --source-file test/sample.c

Write-Host "Full test process completed."
