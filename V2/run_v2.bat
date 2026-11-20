@echo off
chcp 65001 > nul
title AgriShield AI — Plant Disease Specialist App
echo =====================================================================
echo 🌿 กำลังเริ่มต้นระบบ AgriShield AI (Organic Biophilic Edition)...
echo 🌾 Rice (6 คลาส)  -  🌽 Corn (4 คลาส)  -  🥔 Potato (3 คลาส)  -  🍅 Tomato (10 คลาส)
echo 📊 สถาปัตยกรรม: 4 Modular Specialist Models + Grad-CAM + Biocontrol (SDG 12)
echo 🌐 กำลังเปิดเว็บแอปพลิเคชันที่: http://127.0.0.1:8000
echo =====================================================================

cd /d "%~dp0"

:: เคลียร์พอร์ต 8000 และ 8001 เก่าถ้ามีค้าง
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8001 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

:: ตรวจหา Python ในระบบอัตโนมัติ
set "PY_BIN="
where python >nul 2>&1 && set "PY_BIN=python"
if not defined PY_BIN where py >nul 2>&1 && set "PY_BIN=py"
if not defined PY_BIN if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PY_BIN=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PY_BIN if exist "C:\Users\door_\AppData\Local\Programs\Python\Python314\python.exe" set "PY_BIN=C:\Users\door_\AppData\Local\Programs\Python\Python314\python.exe"
if not defined PY_BIN if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY_BIN=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY_BIN if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY_BIN=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY_BIN if exist "%ProgramFiles%\Python314\python.exe" set "PY_BIN=%ProgramFiles%\Python314\python.exe"

if not defined PY_BIN (
    echo ❌ [ERROR] ไม่พบคำสั่ง Python ในเครื่องนี้ กรุณาติดตั้ง Python 3.10 ขึ้นไป
    pause
    exit /b 1
)

echo 🐍 ใช้งาน Python: %PY_BIN%
"%PY_BIN%" server.py 8000
pause
