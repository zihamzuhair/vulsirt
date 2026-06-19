param(
    [string]$ConfigPath = "configs/config.yaml",
    [int]$SampleIndex = 0
)

$ErrorActionPreference = "Stop"

python -m helpers.inspect_b4_vectors --config $ConfigPath --sample-index $SampleIndex
