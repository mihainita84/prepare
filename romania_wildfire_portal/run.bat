@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Run setup_and_run.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
