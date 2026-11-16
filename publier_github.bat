@echo off
setlocal EnableDelayedExpansion
title Publier le code sur GitHub
cd /d "%~dp0"
where git >nul 2>&1 || (
  echo Git n'est pas installe. Installez-le depuis https://git-scm.com/download/win puis relancez ce fichier.
  pause & exit /b 1
)
for /f "delims=" %%n in ('git config --global user.name') do set GNAME=%%n
if "!GNAME!"=="" (
  set /p GNAME=Votre nom pour les commits [Marouane Echhaimi] : 
  if "!GNAME!"=="" set GNAME=Marouane Echhaimi
  git config --global user.name "!GNAME!"
)
for /f "delims=" %%e in ('git config --global user.email') do set GMAIL=%%e
if "!GMAIL!"=="" (
  set /p GMAIL=Votre e-mail GitHub : 
  git config --global user.email "!GMAIL!"
)
if not exist ".git" (
  git init -b main || goto fail
)
git add -A || goto fail
set BIG=
for /f "delims=" %%f in ('git ls-files') do if %%~zf GTR 50000000 (
  echo FICHIER TROP GROS pour GitHub : %%f
  set BIG=1
)
if defined BIG (
  echo.
  echo Deplacez ces fichiers hors du dossier SGMBF ^(ou ajoutez-les a .gitignore^), puis relancez.
  git reset -q
  pause & exit /b 1
)
echo.
echo Fichiers qui seront publies (les donnees, caches et runs sont exclus par .gitignore) :
git status --short
echo.
git diff --cached --quiet && (echo Rien de nouveau a publier.) || (
  git commit -m "Identity-leakage audit of ChaLearn First Impressions V2: code, splits and results" || goto fail
)
git remote get-url origin >nul 2>&1 || (
  echo.
  echo Creez d'abord un depot VIDE sur https://github.com/new  ^(sans README^).
  set /p URL=Collez son adresse ^(ex. https://github.com/Maroxane/CL-FI-V2-Identity-Leakage.git^) : 
  git remote add origin "!URL!" || goto fail
)
git push -u origin main && goto done
echo.
echo Le depot distant contient deja un autre historique.
set /p REP=Remplacer TOUT le contenu du depot GitHub par ce dossier ? (O/N) : 
if /i not "!REP!"=="O" goto fail
git push -u --force origin main || goto fail
:done
echo.
echo Publie : 
git remote get-url origin
pause & exit /b 0
:fail
echo.
echo ECHEC - voir le message ci-dessus.
pause & exit /b 1
