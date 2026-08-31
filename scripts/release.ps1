# Release script for PyPI publishing (PowerShell)
param([switch]$TestPyPI)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Path $PSScriptRoot -Parent
Set-Location $RepoRoot

if (-not (Test-Path dist)) {
    Write-Error "dist/ directory not found. Please run .\scripts\build.ps1 first!"
}

# Load TWINE_API_KEY from .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*TWINE_API_KEY\s*=\s*(.+)\s*$") {
            $env:TWINE_PASSWORD = $matches[1].Trim('"').Trim("'")
            $env:TWINE_USERNAME = "__token__"
            Write-Host "Loaded TWINE_API_KEY from .env" -ForegroundColor Green
        }
    }
}

Write-Host "=== Installing / Upgrading Twine ===" -ForegroundColor Cyan
python -m pip install --upgrade twine

if ($TestPyPI) {
    Write-Host "=== Uploading to TestPyPI ===" -ForegroundColor Yellow
    python -m twine upload --repository testpypi dist/*
} else {
    Write-Host "=== Uploading to PyPI Production ===" -ForegroundColor Green
    python -m twine upload dist/*
}
