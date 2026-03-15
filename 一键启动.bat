@echo off
:: 切换命令行编码为UTF-8，修复中文乱码
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动膝关节康复角度波形分类系统...
echo 正在加载AI模型(PyTorch)... 这可能需要10-15秒，请耐心等待。
echo 请保持此窗口开启。关闭窗口将停止服务。

:: 在后台延时8秒后再打开浏览器，给服务器一点启动时间
start /b cmd /c "timeout /t 8 /nobreak >nul && start "" "http://localhost:5002""

:: 启动 Python 后端服务器
python backend_server.py
pause