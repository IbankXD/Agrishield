@echo off
chcp 65001 > nul
title AgriShield AI V2 — Multi-Model Specialist App (Port 8001)
echo =====================================================================
echo 🚀 กำลังเริ่มต้นระบบ AgriShield AI V2 (สถาปัตยกรรม 4 โมเดลผู้เชี่ยวชาญ)...
echo 🌾 Rice (6 คลาส) | 🌽 Corn (4 คลาส) | 🥔 Potato (3 คลาส) | 🍅 Tomato (10 คลาส)
echo 📊 สัดส่วนข้อมูล: Train 70%% : Val 15%% : Test 15%%
echo 🌐 กำลังเปิดเว็บแอปพลิเคชันที่: http://127.0.0.1:8001
echo =====================================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8001 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
cd /d "%~dp0"
start http://127.0.0.1:8001

:: ตรวจหา Python ในระบบอัตโนมัติ
set "PY_BIN="
where python >nul 2>&1 && set "PY_BIN=python"
if not defined PY_BIN where py >nul 2>&1 && set "PY_BIN=py"
if not defined PY_BIN if exist "C:\Users\door_\AppData\Local\Programs\Python\Python314\python.exe" set "PY_BIN=C:\Users\door_\AppData\Local\Programs\Python\Python314\python.exe"

if not defined PY_BIN (
    echo ❌ [ERROR] ไม่พบคำสั่ง Python ในเครื่องนี้ กรุณาติดตั้ง Python 3.10 ขึ้นไป
    pause
    exit /b 1
)

echo 🐍 ใช้งาน Python: %PY_BIN%
"%PY_BIN%" server.py 8001
pause
