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
"C:\Users\door_\AppData\Local\Programs\Python\Python314\python.exe" server.py 8001
pause
