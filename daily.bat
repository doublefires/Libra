@echo off
chcp 65001 >nul
set "BAROMETER_DATA_DIR=%~dp0data_real"
set "BAROMETER_OUTPUT_DIR=%~dp0outputs_real"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
python scripts\daily.py %*
python scripts\paper.py --mode v8
python scripts\paper.py --mode rotation
pause
