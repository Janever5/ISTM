@echo off
chcp 65001 >nul
echo ================================================
echo   Janus Sensor 论文出图平台 V2
echo ================================================
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit
)
if not exist ".installed" (
    echo [首次运行] 正在安装依赖库...
    pip install -r requirements.txt
    echo. > .installed
)
echo 正在启动... 浏览器打开 http://127.0.0.1:5004
start http://127.0.0.1:5004
python app.py
pause