@echo off
title SG-MBF - reparation des dependances
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environnement absent : lancez start.bat
  pause
  exit /b 1
)
echo.
echo  Installation forcee de pyarrow 17.0.0 (compatible NumPy 1.26, requis par MediaPipe) ...
echo.
".venv\Scripts\python.exe" -m pip install "pyarrow==17.0.0" --no-deps --force-reinstall
echo.
".venv\Scripts\python.exe" -c "import numpy, pyarrow, opensmile, mediapipe; print('OK  numpy', numpy.__version__, '| pyarrow', pyarrow.__version__, '| opensmile et mediapipe importes')"
echo.
pause
