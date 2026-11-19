@echo off
setlocal
title Stop PkmLens Services
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=D:\Project\CardLens\.venv-yolo\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" scripts\serve.py --stop
pause
