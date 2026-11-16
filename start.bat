@echo off
setlocal EnableDelayedExpansion
title SG-MBF - demarrage
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
echo.
echo  ==============================================================
echo   SG-MBF  -  Shared-Gate Multi-Branch Fusion (ChaLearn FI V2)
echo  ==============================================================
echo.

if exist ".venv\installed.flag" goto run

echo  [1/4] Recherche de Python 3.10 / 3.11 / 3.12 ...
set "BASEPY="
for %%V in (3.11 3.12 3.10) do (
  if not defined BASEPY (
    py -%%V -c "import sys" >nul 2>&1 && set "BASEPY=py -%%V"
  )
)
if not defined BASEPY (
  python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
  echo.
  echo  ERREUR : Python 3.10, 3.11 ou 3.12 est requis ^(MediaPipe ne supporte pas 3.13^).
  echo  Installez Python 3.11 depuis https://www.python.org/downloads/ puis relancez start.bat
  pause
  exit /b 1
)
echo        Python trouve : !BASEPY!

if not exist "%PY%" (
  echo  [2/4] Creation de l'environnement virtuel .venv ...
  !BASEPY! -m venv .venv || goto fail
)
"%PY%" -m pip install --upgrade pip --quiet

echo  [3/4] Installation de PyTorch 2.5.1 + CUDA 12.4 ^(environ 2.5 Go^) ...
"%PY%" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || goto fail

echo  [4/4] Installation des autres dependances ...
"%PY%" -m pip install -r requirements.txt || goto fail
"%PY%" -m pip install "pyarrow==17.0.0" --no-deps --force-reinstall || goto fail
echo ok> ".venv\installed.flag"
echo.
echo  Installation terminee.
echo.

:run
if not exist logs mkdir logs
echo  Demarrage des interfaces web ...
"%PY%" launch.py start
if errorlevel 1 goto fail
echo.
echo  Les interfaces tournent en arriere-plan. Vous pouvez fermer cette fenetre.
echo  Pour tout arreter : double-cliquez sur stop.bat
echo.
timeout /t 8 >nul
exit /b 0

:fail
echo.
echo  ECHEC. Consultez les messages ci-dessus et le dossier logs\
pause
exit /b 1
