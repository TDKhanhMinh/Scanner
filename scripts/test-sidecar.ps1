<#
.SYNOPSIS
    Smoke-test a packaged Attendance Scanner sidecar without Python.

.DESCRIPTION
    Runs --version, --help, plan, and scan-batch against an input root that
    contains representative JPG, PNG, and WEBP files. Scanner state is isolated
    in a temporary APPDATA directory.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputRoot,
    [string]$SidecarPath = "",
    [string]$OutputRoot = "",
    [ValidateRange(1, 3600)]
    [int]$TimeoutSeconds = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($SidecarPath)) {
    $SidecarPath = Join-Path $repoRoot "apps\desktop\src-tauri\binaries\attendance-scanner-sidecar-x86_64-pc-windows-msvc.exe"
}
$InputRoot = (Resolve-Path $InputRoot).Path
$ownsOutputRoot = [string]::IsNullOrWhiteSpace($OutputRoot)
if ($ownsOutputRoot) {
    $OutputRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
        "attendance-scanner-sidecar-smoke-output-" + [Guid]::NewGuid().ToString("N")
    )
}

if (-not (Test-Path -LiteralPath $SidecarPath -PathType Leaf)) {
    throw "Packaged sidecar not found at '$SidecarPath'. Run scripts\build-sidecar.ps1 first."
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Smoke output path already exists; choose a new path to avoid overwriting data: '$OutputRoot'."
}

$requiredExtensions = @(".jpg", ".png", ".webp")
$foundExtensions = @(
    Get-ChildItem -LiteralPath $InputRoot -Recurse -File |
        ForEach-Object { $_.Extension.ToLowerInvariant() } |
        Sort-Object -Unique
)
foreach ($requiredExtension in $requiredExtensions) {
    if ($foundExtensions -notcontains $requiredExtension) {
        throw "Smoke input must contain at least one $requiredExtension file."
    }
}

function Invoke-Sidecar {
    param(
        [string[]]$Arguments,
        [string]$AppDataPath,
        [int]$TimeoutMilliseconds
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $SidecarPath
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.Environment["APPDATA"] = $AppDataPath
    function Quote-WindowsArgument {
        param([string]$Argument)

        if ($Argument -notmatch '[\s"]' -and $Argument.Length -gt 0) {
            return $Argument
        }
        $escaped = [regex]::Replace($Argument, '(\\*)"', '$1$1\"')
        $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
        return '"' + $escaped + '"'
    }

    if ($startInfo.PSObject.Properties.Name -contains "ArgumentList") {
        foreach ($argument in $Arguments) {
            $startInfo.ArgumentList.Add($argument)
        }
    } else {
        # Windows PowerShell 5.1 targets .NET Framework, which does not expose
        # ProcessStartInfo.ArgumentList. Quote each argument for its legacy
        # command-line parser while preserving spaces and Unicode paths.
        $startInfo.Arguments = (($Arguments | ForEach-Object {
            Quote-WindowsArgument ([string]$_)
        }) -join ' ')
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $started = $false
    try {
        if (-not $process.Start()) {
            throw "Unable to start '$SidecarPath'."
        }
        $started = $true
        # Start both async reads before waiting so a full stderr pipe cannot
        # block the child while stdout is still producing JSONL.
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            throw "Sidecar command timed out after $TimeoutMilliseconds ms: $($Arguments -join ' ')"
        }
        # Wait again so both async readers observe EOF before collecting output.
        $process.WaitForExit()
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = $stdoutTask.GetAwaiter().GetResult()
            Stderr = $stderrTask.GetAwaiter().GetResult()
        }
    } finally {
        if ($started -and -not $process.HasExited) {
            $process.Kill($true)
            $process.WaitForExit()
        }
        $process.Dispose()
    }
}

function Read-Events {
    param([string]$Jsonl)

    $events = @()
    foreach ($line in ($Jsonl -split '\r?\n')) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $event = $line | ConvertFrom-Json
        } catch {
            throw "Sidecar emitted invalid JSONL: $line"
        }
        if ($event.protocolVersion -ne 1) {
            throw "Sidecar emitted an unsupported protocolVersion."
        }
        $events += $event
    }
    return $events
}

$smokeAppData = Join-Path ([System.IO.Path]::GetTempPath()) (
    "attendance-scanner-sidecar-smoke-appdata-" + [Guid]::NewGuid().ToString("N")
)

try {
    $timeoutMilliseconds = $TimeoutSeconds * 1000
    $version = Invoke-Sidecar -Arguments @("--version") -AppDataPath $smokeAppData -TimeoutMilliseconds $timeoutMilliseconds
    if ($version.ExitCode -ne 0 -or $version.Stdout.Trim() -ne "attendance-scanner 0.1.0") {
        throw "Packaged --version smoke test failed."
    }

    $help = Invoke-Sidecar -Arguments @("--help") -AppDataPath $smokeAppData -TimeoutMilliseconds $timeoutMilliseconds
    if ($help.ExitCode -ne 0 -or $help.Stdout -notmatch "scan-batch") {
        throw "Packaged --help smoke test failed."
    }

    $plan = Invoke-Sidecar -Arguments @("plan", "--input", $InputRoot, "--output", $OutputRoot) -AppDataPath $smokeAppData -TimeoutMilliseconds $timeoutMilliseconds
    if ($plan.ExitCode -ne 0) {
        throw "Packaged plan smoke test failed: $($plan.Stderr)"
    }
    $planEvents = @(Read-Events $plan.Stdout)
    if ($planEvents.Count -ne 1 -or $planEvents[0].type -ne "scan_plan") {
        throw "Packaged plan did not emit exactly one scan_plan event."
    }
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "Plan smoke test unexpectedly created the output directory."
    }

    $scan = Invoke-Sidecar -Arguments @(
        "scan-batch", "--input", $InputRoot, "--output", $OutputRoot, "--workers", "1"
    ) -AppDataPath $smokeAppData -TimeoutMilliseconds $timeoutMilliseconds
    if ($scan.ExitCode -notin @(0, 2)) {
        throw "Packaged scan-batch smoke test failed with exit code $($scan.ExitCode): $($scan.Stderr)"
    }
    $scanEvents = @(Read-Events $scan.Stdout)
    $eventTypes = @($scanEvents | Select-Object -ExpandProperty type)
    foreach ($requiredType in @("scan_plan", "file_started", "scan_completed")) {
        if ($eventTypes -notcontains $requiredType) {
            throw "Packaged scan-batch is missing the $requiredType event."
        }
    }
    $pdfCount = @(Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter "*.pdf" -File).Count
    if ($pdfCount -lt 3) {
        throw "Packaged scan-batch produced $pdfCount PDFs; expected at least 3 representative outputs."
    }

    $relaunchPlan = Invoke-Sidecar -Arguments @(
        "plan", "--input", $InputRoot, "--output", $OutputRoot
    ) -AppDataPath $smokeAppData -TimeoutMilliseconds $timeoutMilliseconds
    if ($relaunchPlan.ExitCode -ne 0) {
        throw "Packaged relaunch plan smoke test failed: $($relaunchPlan.Stderr)"
    }
    $relaunchEvents = @(Read-Events $relaunchPlan.Stdout)
    if ($relaunchEvents.Count -ne 1 -or $relaunchEvents[0].type -ne "scan_plan") {
        throw "Packaged relaunch did not emit exactly one scan_plan event."
    }
    if ($relaunchEvents[0].filesToProcess -ne 0 -or $relaunchEvents[0].unchanged -lt 3) {
        throw "Packaged relaunch did not preserve unchanged manifest state."
    }
    Write-Host "Packaged sidecar smoke test passed: version/help/plan/scan-batch and $pdfCount PDF outputs." -ForegroundColor Green
} finally {
    if (Test-Path -LiteralPath $smokeAppData) {
        Remove-Item -LiteralPath $smokeAppData -Recurse -Force
    }
    if ($ownsOutputRoot -and (Test-Path -LiteralPath $OutputRoot)) {
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
}
