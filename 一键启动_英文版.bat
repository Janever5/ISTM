@echo off
:: 切换编码为UTF-8
chcp 65001 >nul
cd /d "%~dp0"
echo Starting Knee Rehabilitation System (English Mode)...
echo Loading AI models (PyTorch)... This may take 10-15 seconds.
echo Please keep this window OPEN. Closing it will stop the server.

:: 延时8秒后打开英文版网页
start /b cmd /c "timeout /t 8 /nobreak >nul && start "" "http://localhost:5002?lang=en""

:: 启动服务器
python backend_server.py
pause