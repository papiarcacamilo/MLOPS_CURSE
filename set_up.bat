@echo off
echo ===============================================
echo   Configuracion inicial - MLOPS_CURSE
echo ===============================================

echo.
echo Instalando dependencias desde requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Configuracion completada. Dependencias instaladas correctamente.
pause
