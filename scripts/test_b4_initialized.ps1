param(
    [string[]]$Configs,
    [ValidateSet("primevul", "rust", "all")]
    [string]$Dataset = "all",
    [switch]$Overwrite
)

. "$PSScriptRoot\test_common.ps1"

function Get-ConfigCheckpointDir {
    param([string]$ConfigPath)

    $script = @"
from helpers.config_loader import load_config
cfg = load_config(r'$ConfigPath')
print(cfg['paths']['checkpoints'])
"@
    $value = & python -c $script
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read checkpoint directory from $ConfigPath"
    }
    return ($value | Select-Object -Last 1).Trim()
}

function New-InitializedConfig {
    param(
        [string]$ConfigPath,
        [string]$OutputPath,
        [string]$RunName
    )

    $script = @"
from pathlib import Path
import yaml
from helpers.config_loader import load_config

config_path = Path(r'$ConfigPath')
output_path = Path(r'$OutputPath')
run_name = r'$RunName'

cfg = load_config(config_path)
original_checkpoint_dir = cfg['paths']['checkpoints']

cfg.setdefault('model', {}).setdefault('encoder_initialization', {})
cfg['model']['encoder_initialization']['enabled'] = True
cfg['model']['encoder_initialization']['source_checkpoint'] = str(Path(original_checkpoint_dir) / 'b1_best.pt').replace('\\', '/')
cfg['model']['encoder_initialization']['ir_checkpoint'] = str(Path(original_checkpoint_dir) / 'b2_best.pt').replace('\\', '/')

cfg['paths']['checkpoints'] = f'E:/Checkpoints/{run_name}_b4_initialized'
cfg['paths']['results'] = f'results/{run_name}_b4_initialized'
cfg['paths']['logs'] = f'logs/{run_name}_b4_initialized'
cfg['training']['baseline'] = 'b4'
cfg['training']['resume'] = False

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open('w', encoding='utf-8') as file:
    yaml.safe_dump(cfg, file, sort_keys=False)
print(output_path)
"@
    & python -c $script
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate initialized B4 config for $ConfigPath"
    }
}

Enter-ProjectRoot
try {
    $configList = Resolve-ConfigList $Configs
    $generatedDir = "temp/generated_configs"
    New-Item -ItemType Directory -Force -Path $generatedDir | Out-Null

    foreach ($config in $configList) {
        Assert-ProjectPath $config "config"
        $runName = Get-RunName $config
        $checkpointDir = Get-ConfigCheckpointDir $config
        $b1Checkpoint = Join-Path $checkpointDir "b1_best.pt"
        $b2Checkpoint = Join-Path $checkpointDir "b2_best.pt"

        if (-not (Test-Path -LiteralPath $b1Checkpoint) -or -not (Test-Path -LiteralPath $b2Checkpoint)) {
            Write-Host "Skipping initialized B4 for $config because B1/B2 checkpoints are missing."
            Write-Host "  Expected: $b1Checkpoint"
            Write-Host "  Expected: $b2Checkpoint"
            continue
        }

        $generatedConfig = Join-Path $generatedDir "$($runName)_b4_initialized.yaml"
        New-InitializedConfig $config $generatedConfig $runName

        Write-Host "Training initialized B4 with $generatedConfig"
        Invoke-ProjectPython -Arguments @("train.py", "--config", $generatedConfig, "--baseline", "b4")

        Write-Host "Evaluating initialized B4 with $generatedConfig on $Dataset"
        $evalArgs = @("evaluate.py", "--config", $generatedConfig, "--baseline", "b4", "--dataset", $Dataset)
        if ($Overwrite) {
            $evalArgs += "--overwrite"
        }
        Invoke-ProjectPython -Arguments $evalArgs
    }
}
finally {
    Pop-Location
}
