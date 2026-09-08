<#
.SYNOPSIS
    Sets up local Python virtual environment and installs Node dependencies for Attendance Scanner.
#>

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Setting up Python virtual environment in scanner/.venv ===" -ForegroundColor Cyan
if (!(Test-Path "scanner\.venv")) {
    python -m venv scanner\.venv
}

$venvPython = "scanner\.venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -e "scanner[dev]"
& $venvPython -m pip install -r "scanner\requirements-sidecar.lock"

Write-Host "=== 2. Setting up Desktop dependencies in apps/desktop ===" -ForegroundColor Cyan
Push-Location "apps\desktop"
try {
    npm install
} finally {
    Pop-Location
}

Write-Host "=== Setup completed successfully! ===" -ForegroundColor Green
