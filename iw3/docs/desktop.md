# iw3-desktop: PC Desktop 3D Streaming Tool

[Japanese Description:日本語の説明](desktop_ja.md)

## Overview

`iw3-desktop` is a tool that converts your PC desktop screen into 3D in real-time and streams it over Wi-Fi. It can be viewed as side-by-side 3D from the browser on VR devices like Meta Quest.

You can watch any image and video displayed on your PC in real-time within a VR space. Note that `iw3-desktop` itself streams video only (without audio or remote input):

*   **Standalone VR Browser (e.g., Meta Quest Browser)**: To control the PC, you continue to use the PC's keyboard and mouse, and audio is output from the PC (speakers/headphones). The VR device functions solely as a display.
*   **PCVR (e.g., SteamVR / Local Viewer with VR desktop software)**: Audio is output through the PC's configured audio device. In PCVR environments such as SteamVR, PC audio is normally routed to the VR headset automatically (unless configured to use PC speakers/headphones). Input/control behavior depends on the PCVR environment or VR desktop software being used.

Please be aware of 3D sickness, as the depth estimation results for GUI windows and text may not be perfect. This tool is primarily intended for full-screen playback of images and videos.

In addition to web streaming, it also supports local viewer window display. This feature utilizes the streaming capabilities of third-party 3D monitors or VR desktop software (such as Virtual Desktop or Bigscreen) that can capture windows or virtual monitors.

## Known Issues

*   Confirmed to work with Meta Quest and PICO 4, but not with VisionPro.
*   On Linux, screen capture depends on the display server (see `Screenshot Method` below):
    *   **Wayland**: Use `--screenshot obs` or `--screenshot kwcapture` (KDE Plasma only).
    *   **X11**: Use `--screenshot mss`.

## Security Notice

`iw3-desktop` starts an HTTPS server (or HTTP when using `--legacy-stream`) without a password by default. Please be aware that other PCs within the same network may be able to access it.

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
Open https://192.168.11.6:1303
Estimated FPS = 30.24, Streaming FPS = 0.00
```
(The `192.168.11.6` address will vary depending on your network environment. When `--legacy-stream` is specified, an `http://` URL is displayed.)

If a Firewall dialog appears, please allow access.

Open the displayed URL in your PC browser or VR device browser and check if the stream can be played.

If the PC's IP address is not detected correctly within the LAN, you can specify the IP address with the `--bind-addr` option:

```bash
python -m iw3.desktop --bind-addr 192.168.1.2
```

## How to View on VR Devices

### WebXR (Default: `Web Streaming`)

By default, `iw3-desktop` uses the WebXR player (`iw3-player`) over HTTPS.

#### About SSL Warnings

To meet WebXR's mandatory HTTPS requirement, the server generates a self-signed certificate on startup.
During your first visit, your browser will display a security warning (e.g., "Your connection is not private"). To proceed:

1.  Click **"Advanced"**.
2.  Select **"Proceed to [IP address]"**.

**Tip:** Bookmarking this page will allow you to skip this warning in the future.

#### Playback Steps

1.  Launch the browser on your VR device (or a WebXR-compatible PC browser for PCVR) and open the `https://...` URL.
2.  Skip the SSL warning if it appears.
3.  Wait for the stream to start, then click **Enter VR** to enter WebXR mode.

For WebXR player controls and screen settings, see the [iw3-player documentation](../player/README.md).

### Web Streaming (Legacy) (`--legacy-stream`)

If you prefer to use the VR browser's built-in HTML5 `<video>` 3D display mode (instead of WebXR), select **`Web Streaming (Legacy)`** in the GUI or pass the `--legacy-stream` option in the CLI:

```bash
python -m iw3.desktop --legacy-stream
```

Confirmed to work with Meta Quest 2 Browser and Google Chrome (does not work on Firefox). On Meta Quest, you can play 3D video by following these steps:

1.  Launch `Browser`.
2.  Enter the `http://...` server URL.
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

You can specify the maximum vertical resolution of the screen with the `--stream-height` option. The default is 1080px (`2160`, `1080`, and `720` are selectable in the GUI). Screens with height less than or equal to this value are streamed without resizing.

```bash
python -m iw3.desktop --stream-height 720
```

### FPS

You can specify the streaming frame rate (FPS) with the `--stream-fps` option. The default is 15 FPS.

```bash
python -m iw3.desktop --stream-fps 30
```

If the `Estimated FPS` is significantly lower than the specified FPS, it means that the PC's performance is not sufficient to process the specified FPS.

Due to `--batch-size 1` processing, the FPS will be considerably lower than during video conversion.

### Video Codec & Quality Settings

The default `Web Streaming` (WebXR) mode supports low-latency H.264 / HEVC video streaming as well as MJPEG (`jpeg` / `jpeg-gpu`). Select the encoder with `--video-codec` (or `Video Codec` in the GUI; default is `libx264`, or `libopenh264` in LGPL builds).

```bash
python -m iw3.desktop --video-codec h264_nvenc
```

*   **Supported codecs**: `libx264`, `libopenh264`, `h264_nvenc`, `hevc_nvenc`, `h264_vaapi`, `hevc_vaapi`, `h264_qsv`, `hevc_qsv`, `h264_amf`, `hevc_amf`, `libx265`, `jpeg`, `jpeg-gpu`
*   **Recommended codec (H.264)**: While the Meta Quest browser supports H.265 (HEVC) decoding, typical PC (PCVR) browsers often only support H.264 for WebCodecs and may fall back to MJPEG when using `hevc_nvenc` / HEVC. Unless you specifically need > 4096 px width, **H.264 (`h264_nvenc`, `libx264`, etc.) is generally recommended**.
*   **Automatic fallback**: If the PC encoder fails or the client browser/device does not support the selected video codec, playback automatically falls back to MJPEG. You can see the active codec next to the name in the player (e.g., `Desktop (1920x1080, hevc_nvenc)` or `Desktop (1920x1080, mjpeg)`).
*   **`Web Streaming (Legacy)`**: Legacy mode only supports MJPEG, so only `jpeg` / `jpeg-gpu` are selectable in the GUI.

#### Resolution Compatibility (H.264 vs HEVC)

H.264 hardware encoders (such as `h264_nvenc`) and typical H.264 decoders are limited to **4096 px** in width/height. When streaming wide Full SBS formats exceeding 4096 px (such as `2560x1080` Full SBS or 4K Full SBS), **HEVC** (`hevc_nvenc`, etc.) is required:

| Screen / Stereo Mode | Stream Resolution | H.264 (`h264_nvenc`, etc.) | HEVC (`hevc_nvenc`, etc.) |
| --- | --- | --- | --- |
| `1920x1080` Half SBS | `1920x1080` | Supported | Supported |
| `1920x1080` Full SBS | `3840x1080` | Supported | Supported |
| `2560x1080` Half SBS | `2560x1080` | Supported | Supported |
| `2560x1080` Full SBS | `5120x1080` | Not supported (> 4096 px) | Supported |
| 4K (`3840x2160`) Half SBS | `3840x2160` | Supported | Supported |
| 4K (`3840x2160`) Full SBS | `7680x2160` | Not supported (> 4096 px) | Supported (falls back to MJPEG if client decoder limit is exceeded) |

#### Stream Quality (`--stream-quality`)

You can specify the video/JPEG quality with the `--stream-quality` option (1-100).

```bash
python -m iw3.desktop --stream-quality 80
```
The default is 90. Specifying a lower value reduces network traffic.

### Screenshot Method (Screen Capture)

`--screenshot` selects the screen capture backend.

| method | platform | monitor / window selection | notes |
| --- | --- | --- | --- |
| `pil` | all | no | `PIL.ImageGrab`. Uses XWayland on Wayland |
| `mss` | Windows, X11, macOS | monitor (window: Windows/X11) | X11 only on Linux |
| `wc_mp` | Windows | monitor + window | `windows_capture` (separate process) |
| `wc_cuda` | Windows + NVIDIA | monitor + window | `wc_cuda` |
| `obs` | Linux Wayland | via Desktop Portal (screen or window) | `libobs` (`linux-pipewire` plugin, separate process) |
| `kwcapture` | Linux Wayland (KDE Plasma) | monitor + window | [kwcapture](https://pypi.org/project/kwcapture/) (`org.kde.KWin.ScreenShot2`). Direct window/monitor selection without portal on KDE |

#### Linux Wayland (`obs` / `kwcapture`)

*   **`obs`**: Uses `libobs` and the `linux-pipewire` plugin via `xdg-desktop-portal`. Monitor or window selection is handled by the system portal dialog at startup.
    *   Required libraries: `libobs`, `obs-plugins` (`linux-pipewire.so`), `libglib-2.0`, `libwayland-client` (e.g. `sudo apt install obs-plugins` on Ubuntu). Note: Flatpak and Snap installations of OBS are not detected.
*   **`kwcapture`**: Uses KWin's `ScreenShot2` D-Bus interface (KDE Plasma only).
    *   Required package: `pip install kwcapture`
    *   The first run installs a small KDE desktop entry so KWin authorizes the capture helper. If a window is missing or the frame is black, check authorization with `python -m kwcapture doctor`.
    *   Sizes are **device** pixels (e.g. on a display scaled to 125%, a 2560 logical px desktop arrives as 3200 px).
    *   `--window-name` matches KWin's window title (caption); dynamic window resizing and maximizing during streaming are supported.
    *   The cursor is composited by KWin, so the extra cursor marker drawn by other backends is not needed (`--disable-draw-cursor` disables it).

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

- `--uncap-fps`: Allows Local Viewer to render at frame rates beyond the display's refresh rate.
- `--fit-window`: Stretches the image to fit the viewer window ignoring aspect ratio.

Even if specified from the CLI, a GUI window will be displayed, so wxpython and OpenGL are required (installed from `requirements-gui.txt`).

On a CUDA run the viewer tries to copy frames straight from GPU memory into the window
(CUDA-GL Interop). That only works when the window is drawn by the same GPU CUDA is running
on, so on a Wayland session whose compositor runs on the integrated GPU, on a PRIME laptop, or
when the graphics driver cannot share buffers, you will see one line like

```
Failed to initialize CUDA-GL Interop: cudaGraphicsGLRegisterBuffer failed: invalid OpenGL or DirectX context (219) [Intel ... / CUDA NVIDIA GeForce RTX 3080 Ti]. Falling back to the CPU transfer path.
```

This is informational: the frames are copied through system memory instead, which is slower but
correct. It is not a Wayland requirement -- capture itself does not use CUDA.
