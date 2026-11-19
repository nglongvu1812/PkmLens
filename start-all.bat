@echo off
setlocal
title PkmLens System Launcher
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=D:\Project\CardLens\.venv-yolo\Scripts\python.exe"
if not exist "%PYTHON%" (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON=python"
    ) else (
        echo [LOI] Khong tim thay Python! Vui long kiem tra moi truong.
        pause
        exit /b 1
    )
)

set "CLOUDFLARED="
where cloudflared >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "CLOUDFLARED=cloudflared"
) else if exist "C:\Program Files (x86)\cloudflared\cloudflared.exe" (
    set "CLOUDFLARED=C:\Program Files (x86)\cloudflared\cloudflared.exe"
) else if exist "C:\Program Files\cloudflared\cloudflared.exe" (
    set "CLOUDFLARED=C:\Program Files\cloudflared\cloudflared.exe"
)

if defined CLOUDFLARED (
    echo [TUNNEL] Dang khoi dong Cloudflare Tunnel cho http://127.0.0.1:8100 ...
    start "PkmLens Cloudflare Tunnel" cmd /k ""%CLOUDFLARED%" tunnel --url http://127.0.0.1:8100"
) else (
    echo [CANH BAO] Khong tim thay cloudflared. Chay local khong co tunnel.
    echo            Cai nhanh bang: winget install --id Cloudflare.cloudflared -e
)

"%PYTHON%" scripts\serve.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [THONG BAO] He thong da dung hoac gap loi.
    pause
)
