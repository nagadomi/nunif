@rem set environment variables

set ROOT_DIR=%~dp0
set NUNIF_DIR=%ROOT_DIR%nunif
set PYTHON_DIR=%ROOT_DIR%python
set MINGIT_DIR=%ROOT_DIR%git
set PATH=C:\Windows\system32;C:\Windows;C:\WINDOWS\System32\WindowsPowerShell\v1.0;%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%MINGIT_DIR%\cmd

@rem clear python related variables
set PYTHONHOME=
set PYTHONPATH=
set PYTHONSTARTUP=
set PYTHONUSERBASE=
set PYTHONEXECUTABLE=
set PIP_TARGET=

@rem git env
set GIT_CONFIG_NOSYSTEM=1

@rem set msvc variables for torch.compile if it is available
set VCVARS2022_PATH=C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat
set VCVARS2019_PATH=C:/Program Files/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat
if exist "%VCVARS2022_PATH%" (
  call "%VCVARS2022_PATH%"
) else (
  if exist "%VCVARS2019_PATH%" (
     call "%VCVARS2019_PATH%"
  )
)

@rem set vslang to English to avoid UnicodeDecodeError
@rem note that this will not work unless the English language package is installed
set VSLANG=1033

@rem clear github token
set GITHUB_TOKEN=
