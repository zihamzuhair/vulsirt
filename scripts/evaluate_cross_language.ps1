param(
    [string]$ConfigPath = "configs/config.yaml",
    [string]$Baseline = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Baseline)) {
    python evaluate.py --config $ConfigPath --dataset all
} else {
    python evaluate.py --config $ConfigPath --baseline $Baseline --dataset all
}
