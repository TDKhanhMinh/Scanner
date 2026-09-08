<#
.SYNOPSIS
    Build and validate the Windows Attendance Scanner PyInstaller sidecar.

.DESCRIPTION
    Produces a onefile console executable and copies it to the Tauri
    target-triple filename under apps/desktop/src-tauri/binaries.
#>

[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$TargetTriple = "x86_64-pc-windows-msvc"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $repoRoot "scanner\.venv\Scripts\python.exe"
}

$specPath = Join-Path $repoRoot "scanner\packaging\attendance-scanner-sidecar.spec"
$lockPath = Join-Path $repoRoot "scanner\requirements-sidecar.lock"
$buildRoot = Join-Path $repoRoot "build\sidecar"
$distRoot = Join-Path $buildRoot "dist"
$workRoot = Join-Path $buildRoot "work"
$binaryRoot = Join-Path $repoRoot "apps\desktop\src-tauri\binaries"
$builtBinary = Join-Path $distRoot "attendance-scanner-sidecar.exe"
$targetBinary = Join-Path $binaryRoot "attendance-scanner-sidecar-$TargetTriple.exe"

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python runtime not found at '$PythonPath'. Run scripts\setup-env.ps1 or pass -PythonPath."
}
if (-not (Test-Path -LiteralPath $specPath -PathType Leaf)) {
    throw "PyInstaller spec not found at '$specPath'."
}
if (-not (Test-Path -LiteralPath $lockPath -PathType Leaf)) {
    throw "Sidecar dependency lock not found at '$lockPath'."
}

$pythonVersion = (& $PythonPath -c "import sys; print('.'.join(map(str, sys.version_info[:3])))").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersion -ne "3.11.9") {
    throw "Sidecar builds require Python 3.11.9; found '$pythonVersion'."
}

Write-Host "Checking locked sidecar dependencies..." -ForegroundColor Cyan
function Normalize-PackageName {
    param([string]$Name)

    return $Name.ToLowerInvariant().Replace("-", "_").Replace(".", "_")
}

$installedPackages = @{}
$freezeOutput = @(& $PythonPath -m pip freeze)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect installed Python dependencies."
}
foreach ($line in $freezeOutput) {
    if ($line -match '^(?<name>[A-Za-z0-9_.-]+)==(?<version>[^;]+)$') {
        $installedPackages[(Normalize-PackageName $Matches.name)] = $Matches.version
    }
}
foreach ($line in Get-Content -LiteralPath $lockPath) {
    $locked = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($locked) -or $locked.StartsWith("#")) {
        continue
    }
    if ($locked -notmatch '^(?<name>[A-Za-z0-9_.-]+)==(?<version>[^;]+)$') {
        throw "Invalid lock entry '$locked' in '$lockPath'."
    }
    $packageName = Normalize-PackageName $Matches.name
    $expectedVersion = $Matches.version
    if (-not $installedPackages.ContainsKey($packageName)) {
        throw "Locked dependency '$locked' is not installed."
    }
    if ($installedPackages[$packageName] -ne $expectedVersion) {
        throw "Locked dependency '$($Matches.name)' requires $expectedVersion but found $($installedPackages[$packageName])."
    }
}

Write-Host "Checking PyInstaller..." -ForegroundColor Cyan
& $PythonPath -m PyInstaller --version
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not available in '$PythonPath'."
}

New-Item -ItemType Directory -Force -Path $distRoot, $workRoot, $binaryRoot | Out-Null

Write-Host "Building onefile sidecar..." -ForegroundColor Cyan
$pyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--distpath", $distRoot,
    "--workpath", $workRoot,
    $specPath
)
& $PythonPath @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path -LiteralPath $builtBinary -PathType Leaf)) {
    throw "PyInstaller completed without producing '$builtBinary'."
}

Write-Host "Validating packaged CLI..." -ForegroundColor Cyan
$versionOutput = (& $builtBinary --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $versionOutput -ne "attendance-scanner 0.1.0") {
    throw "Packaged --version check failed. Output: '$versionOutput'"
}
$helpOutput = (& $builtBinary --help | Out-String)
if ($LASTEXITCODE -ne 0 -or $helpOutput -notmatch "scan-batch") {
    throw "Packaged --help check failed or scan-batch is missing."
}

Copy-Item -LiteralPath $builtBinary -Destination $targetBinary -Force
$hash = (Get-FileHash -LiteralPath $targetBinary -Algorithm SHA256).Hash
$sizeMiB = [Math]::Round((Get-Item -LiteralPath $targetBinary).Length / 1MB, 2)

Write-Host "Sidecar build completed successfully." -ForegroundColor Green
Write-Host "Artifact: $targetBinary"
Write-Host ("Size: {0} MiB" -f $sizeMiB)
Write-Host "SHA256: $hash"
