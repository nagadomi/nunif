@echo off

setlocal enabledelayedexpansion
call "%~dp0\setenv.bat"

copy /y "%NUNIF_DIR%\windows_package\setenv.bat" "%~dp0\setenv.bat"
copy /y "%NUNIF_DIR%\windows_package\update.bat" "%~dp0\update.bat"
copy /y "%NUNIF_DIR%\windows_package\install.bat" "%~dp0\install.bat"

copy /y "%NUNIF_DIR%\windows_package\nunif-prompt.bat" "%~dp0\nunif-prompt.bat"
copy /y "%NUNIF_DIR%\windows_package\iw3-gui.bat" "%~dp0\iw3-gui.bat"
copy /y "%NUNIF_DIR%\windows_package\iw3-desktop-gui.bat" "%~dp0\iw3-desktop-gui.bat"
copy /y "%NUNIF_DIR%\windows_package\waifu2x-gui.bat" "%~dp0\waifu2x-gui.bat"
copy /y "%NUNIF_DIR%\windows_package\waifu2x-web.bat" "%~dp0\waifu2x-web.bat"

pause
exit /b 0
