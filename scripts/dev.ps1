param([string]$Python = "", [switch]$FrontendOnly)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!$Python) {
    $Python = if (Test-Path ".venv/Scripts/python.exe") { (Resolve-Path ".venv/Scripts/python.exe").Path } else { "D:\Project\CardLens\.venv-yolo\Scripts\python.exe" }
}
$env:PYTHONPATH = "services/core-api;services/ai;."
$env:DATA_DIR = Join-Path (Get-Location) "data"
$env:DATABASE_URL = "sqlite:///" + (Join-Path $env:DATA_DIR "pkmlens.db").Replace("\", "/")
$env:WEB_ORIGIN = "http://127.0.0.1:5173,http://localhost:5173"
$env:AI_URL = "http://127.0.0.1:8101"
$env:STORAGE_MODE = "local"
$env:YOLO_CONFIG_DIR = Join-Path $env:DATA_DIR "yolo-config"
$env:HF_HOME = Join-Path $env:DATA_DIR "huggingface"
$env:AI_INTERNAL_KEY = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
$children = @()
try {
    $ports = if ($FrontendOnly) { @(5173) } else { @(5173,8100,8101) }
    $listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue
    foreach ($port in $ports) { if ($listeners.LocalPort -contains $port) { throw "Port $port is already in use. Use the running PkmLens app or stop its own process first." } }
    if (!$FrontendOnly) {
        & $Python scripts/prepare.py
        if ($LASTEXITCODE -ne 0) { throw "Data preparation failed" }
        foreach ($spec in @(@{Module="cardlens_core.main:app";Port=8100}, @{Module="cardlens_ai.main:app";Port=8101})) {
            $process = Start-Process -PassThru -NoNewWindow -FilePath $Python -ArgumentList @("-m", "uvicorn", $spec.Module, "--host", "127.0.0.1", "--port", $spec.Port)
            $children += @{Process=$process;Module=$spec.Module;Port=$spec.Port}
        }
    }
    npm.cmd run dev
} finally {
    foreach ($entry in $children) {
        $child = $entry.Process
        $workers = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $child.Id -and $_.CommandLine -match [regex]::Escape($entry.Module) -and $_.CommandLine -match "--port $($entry.Port)(\s|$)" }
        foreach ($worker in $workers) { Stop-Process -Id $worker.ProcessId -ErrorAction SilentlyContinue }
        $child.Refresh()
        if (!$child.HasExited) { Stop-Process -Id $child.Id -ErrorAction SilentlyContinue }
    }
}
