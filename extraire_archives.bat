@echo off
title SG-MBF - extraction des archives ChaLearn
cd /d "%~dp0"
if not exist ".venv\installed.flag" (
  echo Lancez d'abord start.bat une fois pour installer l'environnement.
  pause
  exit /b 1
)
echo.
echo  Extraction des archives officielles ChaLearn (mots de passe publies par ChaLearn)
echo  presentes dans le dossier Chalearn ...
echo.
".venv\Scripts\python.exe" -m sgmbf.unzip_chalearn %*
echo.
pause
