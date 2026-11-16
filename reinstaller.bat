@echo off
cd /d "%~dp0"
echo Reinstallation des dependances au prochain demarrage...
del ".venv\installed.flag" >nul 2>&1
echo OK. Lancez start.bat
pause
