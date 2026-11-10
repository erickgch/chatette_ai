@echo off
chcp 65001 >nul
cd /d "%~dp0"

set LOGFILE=%~dp0service.log
echo [%DATE% %TIME%] Starting Chatette service >> "%LOGFILE%"

call .venv\Scripts\activate
echo [%DATE% %TIME%] venv activated >> "%LOGFILE%"

set CUDA_DLLS=%~dp0.venv\Lib\site-packages\nvidia
set PATH=%CUDA_DLLS%\cublas\bin;%CUDA_DLLS%\cudnn\bin;%CUDA_DLLS%\cuda_nvrtc\bin;%PATH%

REM Add Ollama to PATH explicitly in case system PATH not refreshed
set PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%

if not exist .env (
    echo [%DATE% %TIME%] ERROR: .env not found >> "%LOGFILE%"
    exit /b 1
)
echo [%DATE% %TIME%] .env found >> "%LOGFILE%"

ollama list >nul 2>&1
if errorlevel 1 (
    echo [%DATE% %TIME%] Starting Ollama... >> "%LOGFILE%"
    start /min "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 5 /nobreak >nul
) else (
    echo [%DATE% %TIME%] Ollama already running >> "%LOGFILE%"
)

echo [%DATE% %TIME%] Launching api.py >> "%LOGFILE%"
set PYTHONUTF8=1
python core\api.py >> "%LOGFILE%" 2>&1
echo [%DATE% %TIME%] api.py exited >> "%LOGFILE%"
