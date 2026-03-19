@echo off
chcp 65001 >nul
echo.
echo  Stopping Chatette...
echo.

REM Kill process on port 8000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo  Stopping process %%a...
    taskkill /PID %%a /F >nul 2>&1
)

echo  Chatette stopped.
echo.
timeout /t 2 /nobreak >nul