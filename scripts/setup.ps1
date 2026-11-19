param([switch]$Cuda)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 with tkinter is required." }
$Python = (Resolve-Path ".venv/Scripts/python.exe").Path
$Index = if ($Cuda) { "https://download.pytorch.org/whl/cu126" } else { "https://download.pytorch.org/whl/cpu" }
& $Python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url $Index
if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed" }
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python dependencies failed" }
npm.cmd ci --no-fund --no-audit
if ($LASTEXITCODE -ne 0) { throw "Frontend dependencies failed" }
& $Python scripts/prepare.py
if ($LASTEXITCODE -ne 0) { throw "Data preparation failed" }
Write-Host "Ready. Run start-pkmlens.bat for the web app, or start-trainer.bat for local training."
