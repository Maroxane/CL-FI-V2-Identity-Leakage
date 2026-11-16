@echo off
title SG-MBF - organiser le dataset ChaLearn (un seul dossier)
cd /d "%~dp0"
if not exist ".venv\installed.flag" (
  echo Lancez d'abord start.bat une fois pour installer l'environnement.
  pause
  exit /b 1
)
echo.
echo  Regroupe les 10 000 videos ChaLearn dans Chalearn\train, Chalearn\val, Chalearn\test
echo  avec les fichiers annotation_*.pkl et transcription_*.pkl, puis supprime le reste
echo  (hf, official, train-1..6, doublons et tout le dossier "Chalearn V2").
echo  Rien n'est supprime si une seule video etiquetee manque.
echo  Arretez d'abord le projet (stop.bat) et extraire_archives.bat s'ils tournent.
echo.
".venv\Scripts\python.exe" -m sgmbf.organize_chalearn %*
echo.
pause
