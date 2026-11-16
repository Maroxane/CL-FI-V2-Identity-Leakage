@echo off
title SG-MBF - arret
cd /d "%~dp0"
echo.
echo  Arret de SG-MBF (interfaces web et taches en cours) ...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" launch.py stop
) else (
  for %%F in (runs\pids\*.pid) do (
    for /f %%P in (%%F) do taskkill /PID %%P /T /F >nul 2>&1
    del "%%F" >nul 2>&1
  )
)
echo  Termine. Les features en cache et les runs termines sont conserves.
timeout /t 4 >nul
