@echo off
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=D:\Project\CardLens\.venv-yolo\Scripts\python.exe"
"%PYTHON%" desktop\trainer.py
pause
