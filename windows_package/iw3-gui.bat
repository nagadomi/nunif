@echo off

call %~dp0setenv.bat
cd %NUNIF_DIR%
start "" pythonw -m iw3.gui
exit 0
