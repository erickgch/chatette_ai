@echo off
chcp 65001 >nul
echo.
echo  Starting Chatette...
echo.

REM Change to the folder where this bat file lives
cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate

REM Check .env exists
if not exist .env (
    echo  ERROR: .env file not found!
    echo  Please rename env.template.txt to .env and fill in your details.
    echo.
    pause
    exit /b 1
)

REM Start Ollama in background if not running
ollama list >nul 2>&1
if errorlevel 1 (
    echo  Starting Ollama in background...
    start /min ollama serve
    timeout /t 3 /nobreak >nul
)

echo  Chatette is starting on http://localhost:8000
echo  Press Ctrl+C to stop.
echo.

python core\api.py
pause