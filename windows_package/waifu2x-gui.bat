@echo off

call %~dp0setenv.bat
cd %NUNIF_DIR%
start "" pythonw -m waifu2x.gui
exit 0
