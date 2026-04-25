$ErrorActionPreference = "Stop"

$ConfigPath = if ($args.Count -ge 1) { $args[0] } else { "test/test_config.yaml" }

Write-Host "1. Running preprocessing"
python preprocess.py --config $ConfigPath

Write-Host "2. Loading processed dataset splits"
python test/test_data_pipeline.py --config $ConfigPath
