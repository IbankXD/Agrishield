@echo off
chcp 65001 > nul
title AgriShield AI - Web Application (PyTorch Engine)
echo =====================================================================
echo 🌱 กำลังเริ่มต้นระบบ AgriShield AI Web Application (PyTorch Engine)...
echo =====================================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
cd /d "%~dp0\web_app"
start http://127.0.0.1:8000
"C:\Users\door_\AppData\Local\Programs\Python\Python314\python.exe" server.py 8000
pause
