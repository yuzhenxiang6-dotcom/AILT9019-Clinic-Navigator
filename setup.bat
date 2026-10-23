@echo off
REM ---------------------------------------------------------------------------
REM  setup.bat - one-click environment setup for Windows
REM
REM  Creates a virtual environment and installs everything from requirements.txt.
REM  Safe to run more than once.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo.
echo ===========================================================
echo   AILT9019 - UHS Retrieval QA ^| Environment Setup
echo ===========================================================
echo.

REM --- Locate a suitable Python -----------------------------------------------
set "PY="
where py >nul 2>nul && (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if "%PY%"=="" (
    where python >nul 2>nul && (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY=python"
    )
)

if "%PY%"=="" (
    echo [ERROR] Python 3.10 or newer is required but was not found.
    echo         Install it from https://www.python.org/downloads/
    echo         IMPORTANT: tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Using %PY%

REM --- Create the virtual environment ----------------------------------------
if not exist "venv" (
    echo [2/3] Creating virtual environment ^(venv^)...
    %PY% -m venv venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Could not create the virtual environment.
        echo.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Virtual environment already exists, reusing it.
)

REM --- Install dependencies ---------------------------------------------------
echo [3/3] Installing dependencies. This can take several minutes.
echo.
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>nul
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed. Common causes:
    echo         - No internet connection
    echo         - Corporate proxy or firewall blocking PyPI
    echo         Try:  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)

echo.
echo ===========================================================
echo   Setup complete.
echo.
echo   Next:  start.bat              (to run the app)
echo          python run_tests.py    (to run the 12 test cases)
echo.
echo   NOTE:  .env is required for online mode.
echo          Copy .env.example to .env and paste your API key.
echo          Without it the app still runs, in offline mode.
echo ===========================================================
echo.
pause
endlocal