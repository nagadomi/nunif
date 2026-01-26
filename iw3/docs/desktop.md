# iw3-desktop: PC Desktop 3D Streaming Tool

[Japanese Description:日本語の説明](desktop_ja.md)

## Overview

`iw3-desktop` is a tool that converts your PC desktop screen into 3D in real-time and streams it over Wi-Fi. It can be viewed as side-by-side 3D from the browser on VR devices like Meta Quest.

You can watch any image and video displayed on your PC in real-time within a VR space. To control the PC, you will continue to use the PC's keyboard and mouse, and audio will also be output from the PC. The VR device functions solely as a display.

Please be aware of 3D sickness, as the depth estimation results for GUI windows and text may not be perfect. This tool is primarily intended for full-screen playback of images and videos.

In addition to web streaming, it also supports local viewer window display. This feature utilizes the streaming capabilities of third-party 3D monitors or VR desktop software (such as Virtual Desktop or Bigscreen) that can capture windows or virtual monitors.

## Known Issues

*   Confirmed to work with Meta Quest and PICO 4, but not with VisionPro.
*   Performance is significantly degraded in Linux/Wayland environments. For Linux users, X11 environment is recommended.

## Security Notice

`iw3-desktop` starts an HTTP server without a password by default. Please be aware that other PCs within the same network may be able to access it.

You can set a password for Basic Authentication using the `--password` option.

## GUI Usage

If you are using `nunif-windows-package` on Windows, the GUI can be launched by running `iw3-desktop-gui.bat`. If `iw3-desktop-gui.bat` does not exist, please run `update-installer.bat` and then `update.bat`.

To launch from the command line, use the following command:

```bash
python -m iw3.desktop.gui
```

![iw3-desktop-gui](https://github.com/user-attachments/assets/18175b2a-a027-42ce-ae5c-a9ee7ae178e5)

## CLI Usage

Launch the server with the following command:

```bash
python -m iw3.desktop
```
(If you are using `nunif-windows-package` on Windows, launch `nunif-prompt.bat` and enter the above command from the console.)

If the server starts successfully, a message like the following will be displayed:

```
Open http://192.168.11.6:1303
Estimated FPS = 30.24, Streaming FPS = 0.00
```
(The `192.168.11.6` address will vary depending on your network environment.)

If a Firewall dialog appears, please allow access.

Open the displayed URL in your PC browser and check if the video can be played. The web page has been confirmed to work with Google Chrome and Meta Quest 2 Browser. It does not work on Firefox.

If the PC's IP address is not detected correctly within the LAN, you can specify the IP address with the `--bind-addr` option:

```bash
python -m iw3.desktop --bind-addr 192.168.1.2
```

## How to View on Meta Quest

Confirmed to work with Meta Quest 2. You can play 3D videos by following these steps:

1.  Launch `Browser`.
2.  Enter the server URL.
3.  (Optional) Add the URL to favorites.
4.  Play the video.
5.  Make the **Browser** full screen using the icon in the top right corner of the Browser.
6.  Make the **video** full screen using the icon in the bottom right corner of the video.
7.  Set the `Display Mode > 3D Side-by-Side` from the screen icon in the front menu at the bottom of the Browser.
8.  (Optional) Set to Curve Window.

Please note that the display mode can only be changed when both the video and the browser are in full-screen mode.

After that, you can use the PC's keyboard and mouse to operate the displayed screen.

## Options

### PICO 4 Specific Options

According to user reports, PICO 4's browser displays videos in Full SBS mode.

You can change the streaming video to Full SBS using the `--full-sbs` option.

```bash
python -m iw3.desktop --full-sbs
```
The default is Half SBS. Meta Quest's browser only supports Half SBS.

### Resolution (Video Resolution)

You can specify the vertical resolution of the screen with the `--stream-height` option. The default is 1080px.

```bash
python -m iw3.desktop --stream-height 720
```

### FPS

You can specify the streaming frame rate (FPS) with the `--stream-fps` option. The default is 15 FPS.

```bash
python -m iw3.desktop --stream-fps 30
```

If the `Estimated FPS` is significantly lower than the specified FPS, it means that the PC's performance is not sufficient to process the specified FPS.

Due to `--batch-size 1` processing, the FPS will be considerably lower than during video conversion. Also, due to browser limitations, it may not be possible to achieve an FPS higher than `Streaming FPS = 30`.

### MJPEG Settings

You can specify the JPEG quality with the `--stream-quality` option (0-100).

```bash
python -m iw3.desktop --stream-quality 80
```
The default is 90. Specifying a lower value reduces network traffic.

### Stereo Settings

The same options as in GUI/CLI can be specified.

```bash
python -m iw3.desktop --depth-model ZoeD_Any_N --divergence 2 --convergence 0.5 --resolution 518
```

The default is `--depth-model Any_V2_S --divergence 1 --convergence 1`.

### Network

You can specify the address and port to launch the HTTP server with the `--bind-addr` and `--port` options.

```bash
python -m iw3.desktop --port 7860
```

To publish the server on the Internet (**not recommended**):

```bash
python -m iw3.desktop --bind-addr 0.0.0.0 --port 7860
```

### Authentication

You can configure HTTP Basic Authentication with the `--user` and `--password` options.

```bash
python -m iw3.desktop --password iw3
```

```bash
python -m iw3.desktop --user admin --password 1234
```

### Local Viewer

Specify the `--local-viewer` option.

Even if specified from the CLI, a GUI window will be displayed, so wxpython and OpenGL are required (installed from `requirements-gui.txt`).
