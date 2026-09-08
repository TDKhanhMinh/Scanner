<#
.SYNOPSIS
    Build the Windows desktop release in deterministic dependency order.

.DESCRIPTION
    Runs frontend build, sidecar packaging, optional packaged smoke test, then
    Tauri bundling. The generated release manifest records artifact metadata.
#>

[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$TargetTriple = "x86_64-pc-windows-msvc",
    [string]$SmokeTestInputRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$desktopRoot = Join-Path $repoRoot "apps\desktop"
$tauriConfigPath = Join-Path $desktopRoot "src-tauri\tauri.conf.json"
$packagePath = Join-Path $desktopRoot "package.json"
$rootPackagePath = Join-Path $repoRoot "package.json"
$sidecarPath = Join-Path $desktopRoot "src-tauri\binaries\attendance-scanner-sidecar-$TargetTriple.exe"
$bundleRoot = Join-Path $desktopRoot "src-tauri\target\release\bundle"
$manifestRoot = Join-Path $repoRoot "build\windows"
$manifestPath = Join-Path $manifestRoot "release-manifest.json"

function Invoke-Checked {
    param(
        [string]$Command,
        [string[]]$Arguments,
        [string]$Step
    )

    Write-Host $Step -ForegroundColor Cyan
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

function Read-JsonFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required metadata file not found at '$Path'."
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

$tauriConfig = Read-JsonFile $tauriConfigPath
$desktopPackage = Read-JsonFile $packagePath
$rootPackage = Read-JsonFile $rootPackagePath
$expectedVersion = [string]$tauriConfig.version
if ([string]::IsNullOrWhiteSpace($expectedVersion)) {
    throw "Tauri version metadata is empty."
}
foreach ($metadata in @($desktopPackage, $rootPackage)) {
    if ([string]$metadata.version -ne $expectedVersion) {
        throw "Version metadata mismatch; expected '$expectedVersion'."
    }
}
if ([string]$tauriConfig.productName -ne "Attendance Scanner") {
    throw "Unexpected Tauri productName '$($tauriConfig.productName)'."
}
foreach ($icon in @($tauriConfig.bundle.icon)) {
    if (-not (Test-Path -LiteralPath (Join-Path $desktopRoot "src-tauri\$icon") -PathType Leaf)) {
        throw "Configured application icon is missing: '$icon'."
    }
}

$npmCommand = (Get-Command npm -ErrorAction Stop).Source
if ([string]::IsNullOrWhiteSpace($SmokeTestInputRoot)) {
    $smokeInput = ""
} else {
    $smokeInput = (Resolve-Path $SmokeTestInputRoot).Path
}

Invoke-Checked $npmCommand @("--prefix", $desktopRoot, "run", "build") "[1/4] Building frontend"

$sidecarScript = Join-Path $PSScriptRoot "build-sidecar.ps1"
$sidecarArguments = @("-TargetTriple", $TargetTriple)
if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
    $sidecarArguments += @("-PythonPath", $PythonPath)
}
$sidecarInvocationArguments = @("-NoProfile", "-File", $sidecarScript) + $sidecarArguments
Invoke-Checked (Get-Command pwsh -ErrorAction Stop).Source $sidecarInvocationArguments "[2/4] Building Python sidecar"

if (-not (Test-Path -LiteralPath $sidecarPath -PathType Leaf)) {
    throw "Expected target-triple sidecar was not produced at '$sidecarPath'."
}
if (-not [string]::IsNullOrWhiteSpace($smokeInput)) {
    $smokeScript = Join-Path $PSScriptRoot "test-sidecar.ps1"
    Invoke-Checked (Get-Command pwsh -ErrorAction Stop).Source @(
        "-NoProfile", "-File", $smokeScript, "-InputRoot", $smokeInput
    ) "[3/4] Running packaged sidecar smoke/relaunch test"
} else {
    Write-Host "[3/4] Packaged sidecar smoke test skipped (pass -SmokeTestInputRoot to run it)." -ForegroundColor Yellow
}

Invoke-Checked $npmCommand @("--prefix", $desktopRoot, "run", "tauri", "build") "[4/4] Building Tauri Windows bundles"

$installers = @(
    Get-ChildItem -LiteralPath $bundleRoot -Recurse -File |
        Where-Object { $_.Extension -in @(".msi", ".exe") -and $_.Name -notlike "attendance-scanner-sidecar-*" }
)
if ($installers.Count -eq 0) {
    throw "Tauri build completed without producing an MSI or NSIS installer under '$bundleRoot'."
}

New-Item -ItemType Directory -Force -Path $manifestRoot | Out-Null
$installerMetadata = @($installers | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    $relativePath = [System.IO.Path]::GetRelativePath($repoRoot, $_.FullName).Replace("\", "/")
    [ordered]@{
        name = $_.Name
        path = $relativePath
        sizeBytes = $_.Length
        sha256 = $hash
    }
})
$sidecarHash = (Get-FileHash -LiteralPath $sidecarPath -Algorithm SHA256).Hash
$manifest = [ordered]@{
    productName = [string]$tauriConfig.productName
    version = $expectedVersion
    targetTriple = $TargetTriple
    sidecar = [ordered]@{
        path = [System.IO.Path]::GetRelativePath($repoRoot, $sidecarPath).Replace("\", "/")
        sizeBytes = (Get-Item -LiteralPath $sidecarPath).Length
        sha256 = $sidecarHash
    }
    installers = $installerMetadata
    offline = $true
    appDataState = "APPDATA/attendance-scanner/state"
    generatedAtUtc = [DateTime]::UtcNow.ToString("O")
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Windows release build completed successfully." -ForegroundColor Green
Write-Host "Release manifest: $manifestPath"
foreach ($installer in $installers) {
    Write-Host "Installer: $($installer.FullName)"
}
