param(
    [string]$ConfigPath = "configs/config.yaml"
)

$ErrorActionPreference = "Stop"

python preprocess.py --config $ConfigPath --dataset all
