# nunif windows package

nunif-windows-package is a online installer for non-developer windows users.

It installs everything needed to run waifu2x-gui and iw3-gui in a single folder.

waifu2x-gui is a super-resolution software. It provides a good-looking upscaling method.

iw3-gui is a software to convert images and videos into 3D images and 3D video, for photograph (non-anime).
The images and videos you really wanted to watch in VR can be watched on VR device as 3D media.

# Prerequisites

- Windows 7 and greater; Windows 10 or greater recommended. Windows Server 2008 r2 and greater
- Microsoft Visual C++ Redistributable Packages https://aka.ms/vs/16/release/vc_redist.x64.exe

# Download and Install

| Steps                                                                         | Material
| ------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------- 
| 1. Download the ZIP file and place it anywhere you choose. In this example, choose `Document\software`| https://github.com/nagadomi/nunif/releases/download/0.0.0/nunif-windows.zip
| 2. Right-click on the downloaded ZIP file to show its file properties.        | ![zip menu](https://github.com/nagadomi/nunif/assets/287255/244b4617-9926-4cf1-941f-dd1d44fe2e28)
| 3. Check the `Security > Unblock` and click `Apply`.                           | ![unblock zip](https://github.com/nagadomi/nunif/assets/287255/dcace34c-8783-44b8-a3fc-953724d4dceb)
| 4. Right-click on the ZIP file and `Extract All`.                             | ![zip_extract](https://github.com/nagadomi/nunif/assets/287255/6c1d167c-8d36-4ba1-aef1-d08b6e153a04)
| 5. Run `install.bat` found in the extracted files.                            | ![file list](https://github.com/nagadomi/nunif/assets/287255/aa4eeafc-7627-4b6f-9964-bd37bef73652)
| 6. A black window will appear and installation will be performed.             | ![install cmd](https://github.com/nagadomi/nunif/assets/287255/c919987b-3cbc-4c89-985b-64e4b8506df7)
| 7. If everything succeeds, you will see  `Press any key to continue` after `Successfully installed nunif`. Press any key to close the window. | ![success](https://github.com/nagadomi/nunif/assets/287255/c49b79b0-f1bd-414e-b311-a50866f34a02)

Installation is now complete.

## waifu2x-gui

Run `waifu2x-gui.bat`.
The first time you run it, it may take long time before the window pops up.

## iw3

Run `iw3-gui.bat`.
The first time you run it, it may take long time before the window pops up.

## Update

Running `update.bat` will update the source code to the latest version on github.

Note that any local changes to the source code may force resets.

If you want to reconfigure, delete the `python`,`git`,`nunif` folders and run `update.bat`.

Running `update-installer.bat` will update `update.bat` itself.

## Factory reset

GUI keeps the input state. If you want to reset it, delete the `nunif\tmp` folder.

## Uninstall

Delete the entire folder.

## Blocked by Windows Defender SmartScreen

![smart screen 1](https://github.com/nagadomi/nunif/assets/287255/66b04d92-695f-4a0e-8db2-6a3cc03a2217)

![smart screen 2](https://github.com/nagadomi/nunif/assets/287255/55fb415e-49c5-440d-977a-a98b2be9e453)

If the alert is raised by Windows, click on **More Info** to display the `Run` button. If you run it once, the alert will not appear the next time.

This alert seems to be displayed for batch files downloaded from the Internet. 
The zip file security unblock step above is an operation to grant this permission in advance.

The batch file is a text file, so you can view its contents in notepad or other editor to verify its safety.
`update.bat`(`install.bat`) does the following.

- Download Embedded Python from the official Python website and extract (in a folder) 
- Download the minimum configuration of Git from `Github for Windows` and extract it (in a folder)
- Download nunif (this program) repository from Github (in a folder)
- Install Python dependencies (in folder)
- Download pre-trained model files (in a folder)

These actions do not intentionally access outside the folder. 

