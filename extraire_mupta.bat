@echo off
title SG-MBF - extraction du corpus MuPTA
cd /d "%~dp0"
if not exist ".venv\installed.flag" (
  echo Lancez d'abord start.bat une fois pour installer l'environnement.
  pause
  exit /b 1
)
echo.
echo  Extraction de MuPTA : Mupta\audio\speaker_XX.zip et Mupta\video\speaker_XX.zip
echo  vers Mupta\extracted\  (environ 50 Go pour la video, verifiez l'espace disque).
echo  Shared.zip est seulement inspecte. Options : --audio-only  --list  --shared
echo.
".venv\Scripts\python.exe" -m sgmbf.unzip_mupta %*
echo.
pause
