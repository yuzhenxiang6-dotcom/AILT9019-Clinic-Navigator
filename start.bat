@echo off
REM ---------------------------------------------------------------------------
REM  start.bat - launch the Streamlit app (Windows)
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not exist "venv" (
    echo.
    echo [ERROR] No virtual environment found.
    echo         Run  setup.bat  first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist ".env" (
    echo.
    echo [NOTE] No .env file found.
    echo        The app will start in OFFLINE MODE and show retrieved text.
    echo        To enable online mode, copy .env.example to .env and add your key.
    echo.
)

echo Starting UHS Retrieval QA at http://localhost:8501
echo Close this window, or press Ctrl+C, to stop the server.
echo.

streamlit run app.py
endlocal