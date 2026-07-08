$ErrorActionPreference = "Stop"

$Script:RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Script:DefaultConfigs = @(
    "configs/100_samples.yaml",
    "configs/250_samples.yaml",
    "configs/500_samples.yaml",
    "configs/1000_samples.yaml"
)
$Script:DefaultBaselines = @("b1", "b2", "b3", "b4")

function Enter-ProjectRoot {
    Push-Location $Script:RepoRoot
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ("python " + ($Arguments -join " "))
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python command failed with exit code $LASTEXITCODE"
    }
}

function Assert-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [string]$Description = "path"
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing ${Description}: $Path"
    }
}

function Get-RunName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ConfigPath
    )

    return [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
}

function Resolve-ConfigList {
    param([string[]]$Configs)

    if ($Configs -and $Configs.Count -gt 0) {
        return $Configs
    }
    return $Script:DefaultConfigs
}

function Resolve-BaselineList {
    param([string[]]$Baselines)

    if ($Baselines -and $Baselines.Count -gt 0) {
        return $Baselines
    }
    return $Script:DefaultBaselines
}
