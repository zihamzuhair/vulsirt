param(
    [string]$ConfigPath = "test/cross_language_config.yaml"
)

$ErrorActionPreference = "Stop"

Write-Host "1. Preprocessing PrimeVul and Rust datasets"
python preprocess.py --config $ConfigPath --dataset all

Write-Host "2. Training B4 on PrimeVul C/C++ samples"
python train.py --config $ConfigPath --baseline b4

Write-Host "3. Evaluating B4 on PrimeVul test and Rust test"
python evaluate.py --config $ConfigPath --baseline b4 --dataset all

Write-Host "4. Saved metrics"
python -c "import json, yaml, pathlib; cfg=yaml.safe_load(open('$ConfigPath', encoding='utf-8')); results=pathlib.Path(cfg['paths']['results']); [print(p, json.dumps(json.load(open(p, encoding='utf-8')), indent=2), sep='\n') for p in sorted(results.glob('b4*_metrics.json'))]"
