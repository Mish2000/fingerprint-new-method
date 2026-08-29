[CmdletBinding()]
param(
    [switch]$Recreate,
    [switch]$FromIntent
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$environmentPrefix = Join-Path $projectRoot ".conda-env"
$intentFile = Join-Path $projectRoot "environment.yml"
$lockFile = Join-Path $projectRoot "conda-lock.yaml"
$condaCandidates = @()
if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE)) {
    $condaCandidates += $env:CONDA_EXE
}
$condaCommand = Get-Command conda.exe -ErrorAction SilentlyContinue
if ($condaCommand) {
    $condaCandidates += $condaCommand.Source
}
$condaCandidates += @(
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\Miniconda3\Scripts\conda.exe"
)
$condaExe = $condaCandidates |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
    Select-Object -Unique |
    Select-Object -First 1

if (-not $condaExe) {
    throw "Conda was not found through CONDA_EXE, PATH, or the supported ProgramData fallbacks."
}

$resolvedProjectRoot = [System.IO.Path]::GetFullPath($projectRoot).TrimEnd('\')
$resolvedPrefix = [System.IO.Path]::GetFullPath($environmentPrefix).TrimEnd('\')
if (-not $resolvedPrefix.StartsWith($resolvedProjectRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to manage an environment outside the project root: $resolvedPrefix"
}

if ($Recreate -and (Test-Path -LiteralPath $environmentPrefix)) {
    & $condaExe remove `
        --prefix $environmentPrefix `
        --all `
        --override-channels `
        --channel conda-forge `
        --yes
    if ($LASTEXITCODE -ne 0) {
        throw "Conda failed to remove the existing project environment."
    }
}

if (-not (Test-Path -LiteralPath $environmentPrefix)) {
    if (-not $FromIntent -and (Test-Path -LiteralPath $lockFile)) {
        $specification = $lockFile
    }
    else {
        $specification = $intentFile
    }
    & $condaExe create `
        --prefix $environmentPrefix `
        --file $specification `
        --override-channels `
        --channel conda-forge `
        --strict-channel-priority `
        --yes
    if ($LASTEXITCODE -ne 0) {
        throw "Conda failed to create the project environment from $specification."
    }
}

$pythonExe = Join-Path $environmentPrefix "python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "The project Python executable is missing: $pythonExe"
}

& $pythonExe -m pip install --no-build-isolation --no-deps --editable $projectRoot
if ($LASTEXITCODE -ne 0) {
    throw "The editable project installation failed."
}

& $pythonExe -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The installed dependency metadata is inconsistent."
}

& $pythonExe -m pytest
if ($LASTEXITCODE -ne 0) {
    throw "The project smoke tests failed."
}

& (Join-Path $environmentPrefix "Scripts\ruff.exe") check $projectRoot
if ($LASTEXITCODE -ne 0) {
    throw "Ruff checks failed."
}

Write-Host "Environment ready: $environmentPrefix"
