@echo off
title SG-MBF - telechargement du dataset
cd /d "%~dp0"
if not exist ".venv\installed.flag" (
  echo Lancez d'abord start.bat une fois pour installer l'environnement.
  pause
  exit /b 1
)
echo.
echo  Telechargement des clips manquants de ChaLearn First Impressions V2
echo  depuis huggingface.co/datasets/yeray142/first-impressions-v2
echo  (validation + test : environ 12 Go). Reprise automatique si interrompu.
echo.
".venv\Scripts\python.exe" -m sgmbf.download %*
echo.
pause
