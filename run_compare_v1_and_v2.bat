@echo off
chcp 65001 > nul
title AgriShield AI — Launching V1 (Port 8000) and V2 (Port 8001) Side-by-Side
echo =====================================================================
echo 🌿 กำลังเปิด AgriShield AI ทั้งสองเวอร์ชันสำหรับนำเสนอและเปรียบเทียบผล:
echo   - V1 (Port 8000): Unified Single Model (23 Classes)
echo   - V2 (Port 8001): Modular 4 Specialist Models (70:15:15 Split)
echo =====================================================================

start "AgriShield V1 (Port 8000)" cmd /c "cd /d %~dp0web_app && run_app.bat"
timeout /t 2 /nobreak > nul
start "AgriShield V2 (Port 8001)" cmd /c "cd /d %~dp0V2 && run_v2.bat"

echo.
echo ✅ เปิดทั้งสองระบบเรียบร้อยแล้ว:
echo    V1: http://127.0.0.1:8000
echo    V2: http://127.0.0.1:8001
echo =====================================================================
timeout /t 5
