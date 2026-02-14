@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PY_CMD="
set "VENV_DIR="

where py >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"

if not defined PY_CMD (
    where python >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [ERROR] Python was not found on this machine.
    echo Install Python 3.9+ from:
    echo   https://www.python.org/downloads/windows/
    echo During setup, enable:
    echo   - Add python.exe to PATH
    echo   - pip
    pause
    endlocal & exit /b 1
)

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 9) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.9+ is required.
    echo Update Python from:
    echo   https://www.python.org/downloads/windows/
    pause
    endlocal & exit /b 1
)

%PY_CMD% -m pip --version >nul 2>nul
if errorlevel 1 (
    echo pip is missing. Attempting to install pip...
    %PY_CMD% -m ensurepip --upgrade >nul 2>nul
)

%PY_CMD% -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] pip is not available.
    echo Reinstall Python 3.9+ with pip enabled from:
    echo   https://www.python.org/downloads/windows/
    pause
    endlocal & exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "VENV_DIR=.venv"
) else if exist "venv\Scripts\python.exe" (
    set "VENV_DIR=venv"
) else (
    echo Creating virtual environment in .venv...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        endlocal & exit /b 1
    )
    set "VENV_DIR=.venv"
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment: %VENV_DIR%
    pause
    endlocal & exit /b 1
)

if exist "requirements.txt" (
    python -m pip install --upgrade pip >nul 2>nul
    python -c "import importlib.util,sys;mods=['streamlit','pandas','numpy','sklearn','plotly','joblib','openpyxl','tabulate'];sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Installing required libraries from requirements.txt...
        python -m pip install -r requirements.txt
        if errorlevel 1 (
            echo [ERROR] Could not install required libraries.
            echo Check your internet connection and try again.
            pause
            endlocal & exit /b 1
        )
    )
) else (
    echo [WARNING] requirements.txt not found. Running with current environment.
)

python -m streamlit run app.py
set "APP_EXIT=%ERRORLEVEL%"
endlocal & exit /b %APP_EXIT%
