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

"%PYTHON%" scripts\serve.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [THONG BAO] He thong da dung hoac gap loi.
    pause
)
