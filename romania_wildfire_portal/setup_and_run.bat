@echo off
setlocal
cd /d "%~dp0"

echo =====================================================
echo  PREPARE WP4 Wildfire Portal - isolated environment setup
echo =====================================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating isolated .venv ...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Python 3.12 was not found via the Windows py launcher.
        echo Install Python 3.12 and make sure the "py" launcher is available.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Existing .venv found.
)

echo [2/4] Upgrading pip inside .venv ...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [3/4] Installing project packages inside .venv ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
    echo Created .env from template. The supplied project also has an embedded FIRMS fallback key.
)

echo [4/4] Starting Streamlit using the .venv Python ...
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b 0

:fail
echo.
echo Setup failed. The global C:\Python312\Scripts folder was NOT used.
pause
exit /b 1
