# Run Codex locally on Windows PowerShell.
# Usage:  pwsh -ExecutionPolicy Bypass -File .\run_local.ps1
# or just: .\run_local.ps1 from inside C:\Users\amyle\codex

Set-Location -Path $PSScriptRoot

# 1. (One-time) Install dependencies. Skip the line below if already installed.
python -m pip install --quiet Flask==3.0.3 requests==2.31.0 user-agents==2.2.0 nglui==2.7.2

# 2. Set env vars (PowerShell syntax — different from bash).
$env:FLASK_SECRET_KEY = "dev-only-not-secret"
$env:PORT = "8080"

# 3. Start the server. First launch downloads the pickled NeuronDB from GCS
#    (~hundreds of MB) into static/data/<version>/. Subsequent launches are fast.
Write-Host "Starting Codex on http://localhost:$($env:PORT)/surprise" -ForegroundColor Cyan
python -m codex.main
