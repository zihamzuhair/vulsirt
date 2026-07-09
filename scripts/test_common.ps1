$ErrorActionPreference = "Stop"

$Script:RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Script:DefaultConfigs = @(
    "configs/100_samples.yaml",
    "configs/250_samples.yaml",
    "configs/500_samples.yaml",
    "configs/1000_samples.yaml"
)
$Script:DefaultBaselines = @("b1", "b2", "b3", "b4")

function Format-Duration {
    param([double]$Seconds)

    $duration = [TimeSpan]::FromSeconds($Seconds)
    return "{0:00}:{1:00}:{2:00}" -f [int][Math]::Floor($duration.TotalHours), $duration.Minutes, $duration.Seconds
}

function Enter-ProjectRoot {
    Push-Location $Script:RepoRoot
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $commandText = "python " + ($Arguments -join " ")
    Write-Host $commandText

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Write-TimingLine "step_start=$commandText at=$(Get-Date -Format o)"
    & python @Arguments
    $exitCode = $LASTEXITCODE
    $timer.Stop()
    Write-TimingLine ("step_end=$commandText exit_code=$exitCode seconds={0:N3} duration={1}" -f $timer.Elapsed.TotalSeconds, (Format-Duration $timer.Elapsed.TotalSeconds))

    if ($exitCode -ne 0) {
        throw "python command failed with exit code $exitCode"
    }
}

function Write-TimingLine {
    param([string]$Line)

    if ($env:VULSIRT_TIMING_LOG) {
        Add-Content -LiteralPath $env:VULSIRT_TIMING_LOG -Value $Line
    }
}

function Start-TestTiming {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunName
    )

    $resultsDir = Join-Path "results" $RunName
    New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $timingPath = Join-Path $resultsDir "run_timing_$timestamp.txt"
    $previousTimingLog = $env:VULSIRT_TIMING_LOG
    $env:VULSIRT_TIMING_LOG = $timingPath

    Set-Content -LiteralPath $timingPath -Value @(
        "run=$RunName",
        "started_at=$(Get-Date -Format o)"
    )

    return [pscustomobject]@{
        Path = $timingPath
        PreviousTimingLog = $previousTimingLog
        Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    }
}

function Stop-TestTiming {
    param(
        [Parameter(Mandatory = $true)]
        $Timing,
        [string]$Status = "completed"
    )

    $Timing.Stopwatch.Stop()
    Add-Content -LiteralPath $Timing.Path -Value @(
        "finished_at=$(Get-Date -Format o)",
        ("status=$Status"),
        ("total_seconds={0:N3}" -f $Timing.Stopwatch.Elapsed.TotalSeconds),
        ("total_duration={0}" -f (Format-Duration $Timing.Stopwatch.Elapsed.TotalSeconds))
    )
    $env:VULSIRT_TIMING_LOG = $Timing.PreviousTimingLog
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
