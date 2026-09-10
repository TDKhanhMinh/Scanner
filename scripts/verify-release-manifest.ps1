<#
.SYNOPSIS
    Verify hashes and offline metadata in a generated Windows release manifest.
#>

[CmdletBinding()]
param(
    [string]$ManifestPath = "build\windows\release-manifest.json",
    [switch]$RequireV2Model
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestFullPath = (Resolve-Path (Join-Path $repoRoot $ManifestPath)).Path
$manifest = Get-Content -LiteralPath $manifestFullPath -Raw | ConvertFrom-Json

if ([string]$manifest.productName -ne "Attendance Scanner") {
    throw "Unexpected productName in release manifest."
}
if ([bool]$manifest.offline -ne $true) {
    throw "Release manifest is not marked offline."
}
if ([string]$manifest.targetTriple -ne "x86_64-pc-windows-msvc") {
    throw "Unsupported target triple in release manifest."
}

function Resolve-RepoArtifact {
    param([string]$RelativePath)

    if ([string]::IsNullOrWhiteSpace($RelativePath) -or [System.IO.Path]::IsPathRooted($RelativePath)) {
        throw "Manifest artifact path must be relative: '$RelativePath'."
    }
    $candidate = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RelativePath))
    $rootPrefix = $repoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest artifact escapes repository root: '$RelativePath'."
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Manifest artifact is missing: '$candidate'."
    }
    return $candidate
}

function Verify-Artifact {
    param($Artifact)

    $path = Resolve-RepoArtifact ([string]$Artifact.path)
    $displayName = [string]$Artifact.path
    if ($null -ne $Artifact.PSObject.Properties["name"]) {
        $displayName = [string]$Artifact.name
    }
    $item = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ([int64]$Artifact.sizeBytes -ne [int64]$item.Length) {
        throw "Size mismatch for '$path'."
    }
    if ($hash -ne ([string]$Artifact.sha256).ToUpperInvariant()) {
        throw "SHA-256 mismatch for '$path'."
    }
    Write-Host ("Verified {0}: {1} bytes" -f $displayName, $item.Length)
}

function Test-ManifestString {
    param(
        $Object,
        [string]$PropertyName
    )

    return $null -ne $Object.PSObject.Properties[$PropertyName] -and
        -not [string]::IsNullOrWhiteSpace([string]$Object.$PropertyName)
}

Verify-Artifact $manifest.sidecar
foreach ($installer in @($manifest.installers)) {
    Verify-Artifact $installer
}

$hasV2Model = (
    $null -ne $manifest.PSObject.Properties["v2"] -and
    $null -ne $manifest.v2 -and
    $null -ne $manifest.v2.PSObject.Properties["modelArtifact"] -and
    $null -ne $manifest.v2.modelArtifact
)
if ($RequireV2Model) {
    if (-not $hasV2Model) {
        throw "V2 model artifact is not recorded; release remains V2-blocked."
    }
    $modelArtifact = $manifest.v2.modelArtifact
    foreach ($property in @("path", "sizeBytes", "sha256")) {
        if ($null -eq $modelArtifact.PSObject.Properties[$property]) {
            throw "V2 model artifact is missing '$property'."
        }
    }
    if (-not (Test-ManifestString $modelArtifact "path")) {
        throw "V2 model artifact path is empty."
    }
    if ([int64]$modelArtifact.sizeBytes -lt 1) {
        throw "V2 model artifact sizeBytes must be positive."
    }
    if ([string]$modelArtifact.sha256 -notmatch '^[0-9A-Fa-f]{64}$') {
        throw "V2 model artifact sha256 must be a 64-character hexadecimal hash."
    }
    Verify-Artifact $modelArtifact
    if (
        $null -eq $manifest.v2.PSObject.Properties["productionReady"] -or
        [bool]$manifest.v2.productionReady -ne $true
    ) {
        throw "V2 model is not marked productionReady; release remains V2-blocked."
    }
    if (
        $null -eq $manifest.v2.PSObject.Properties["packagedStartupVerified"] -or
        [bool]$manifest.v2.packagedStartupVerified -ne $true
    ) {
        throw "V2 packaged startup is not verified; release remains V2-blocked."
    }
}

Write-Host ("Release manifest verified. V2 model recorded: {0}" -f $hasV2Model) -ForegroundColor Green
