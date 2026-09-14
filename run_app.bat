@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title CRYPTO ML - Web Dashboard Server

echo.
echo ================================================================
echo   CRYPTO ML  -  Web Application Dashboard
echo   Privacy-Preserving GNN on Bitcoin Transaction Graphs
echo ================================================================
echo.

:: Check python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python not found.
    echo   Download from https://python.org and add to PATH.
    pause & exit /b 1
)

:: Check if virtual environment exists, activate if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Auto-install FastAPI and Uvicorn if not present
python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo   Installing web dependencies (FastAPI, Uvicorn)...
    pip install fastapi uvicorn -q --no-warn-script-location
)

:: Launch browser in background after 2 seconds
start /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000"

echo   Starting FastAPI Server on http://127.0.0.1:8000...
echo   (Press Ctrl+C to stop the server)
echo.

python app.py
pause
