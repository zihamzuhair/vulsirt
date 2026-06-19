param(
    [string]$ConfigPath = "test/test_config.yaml",
    [int]$SampleIndex = 0
)

$ErrorActionPreference = "Stop"

Write-Host "1. Preprocessing dataset"
python preprocess.py --config $ConfigPath

Write-Host "2. Training B4 adaptive gated fusion"
python train.py --config $ConfigPath --baseline b4

Write-Host "3. Evaluating B4 adaptive gated fusion"
python evaluate.py --config $ConfigPath --baseline b4

Write-Host "4. Inspecting B4 source/IR projections and gate output"
python -m helpers.inspect_b4_vectors --config $ConfigPath --sample-index $SampleIndex

Write-Host "5. B4 metric file"
python -c "import json, yaml, pathlib; cfg=yaml.safe_load(open('$ConfigPath', encoding='utf-8')); p=pathlib.Path(cfg['paths']['results'])/'b4_metrics.json'; print(p); print(json.dumps(json.load(open(p, encoding='utf-8')), indent=2))"

Write-Host "Gated fusion test and evaluation completed."
