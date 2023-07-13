@echo off

call %~dp0setenv.bat
cd %NUNIF_DIR%
python -m waifu2x.web
pause
