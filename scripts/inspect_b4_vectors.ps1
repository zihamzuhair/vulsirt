param(
    [string]$ConfigPath = "configs/config.yaml",
    [int]$SampleIndex = 0
)

$ErrorActionPreference = "Stop"

python inspect_b4_vectors.py --config $ConfigPath --sample-index $SampleIndex
