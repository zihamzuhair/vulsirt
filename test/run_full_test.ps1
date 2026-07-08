$ErrorActionPreference = "Stop"

# The test dataset already has LLVM IR, so first we only check loading/token shapes.
Write-Host "1. Checking test dataset loading"
python test/test_data_pipeline.py --config test/test_config.yaml

Write-Host "2. Training B4 for one tiny smoke-test epoch"
python train.py --config test/test_config.yaml --baseline b4

Write-Host "3. Evaluating B4"
python evaluate.py --config test/test_config.yaml --baseline b4

Write-Host "4. Scanning test/sample.c"
python scanner.py --config test/test_config.yaml --baseline b4 --source-file test/sample.c

Write-Host "Full test process completed."
