# iw3-player

[日本語](README_ja.md)

iw3-player is a self-hosted, specialized viewing environment for stereoscopic media.  
It allows you to stream media that has been pre-converted to 3D with iw3 from your PC and enjoy it on VR devices through a WebXR application.

The player supports a wide range of operations for stereo media.  
One of its features is to instantly adjust people in photos to life-size.  
Viewing photos at a realistic scale significantly enhances immersion and brings your memories to life in VR.

![Screenshot](https://github.com/user-attachments/assets/708d03bc-da99-4cdb-a8e4-2ec247e7e3a2)

## Operating Environment

- **Backend (Python)**: Confirmed to work on Linux and Windows.
- **Frontend (JavaScript)**: Confirmed to work on Meta Quest 2 browser, PC Google Chrome, and Android smartphones.

## Supported Formats

- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp`
- **Videos**: `.mp4`, `.mkv`, `.webm`.
- **Archives**: `.zip`

Supported video formats depend on the browser. The Meta Quest 2 browser supports all combinations of mp4, mkv, h264, and h265 commonly used in iw3. The PC version of Google Chrome does not support h265.
On Meta Quest 2, 4K videos played smoothly.

The player supports both embedded subtitle streams and external files (`.srt` / `.vtt`). External files named identically to the video will be prioritized. Embedded subtitles are automatically converted to `.vtt` by the backend for seamless streaming.

The stereo format (Side-by-Side, Top-Bottom, VR180, etc.) is automatically detected based on tags in filenames, parent folders, or ZIP archives (e.g., `_LR`, `_TB`, `VR180`).
Even for files without tags, the format is automatically estimated from the image or video content during playback. Any manual adjustments to the format are saved per file.

## How to Start

### Windows

Execute `iw3-player-gui.bat` to launch the graphical user interface (GUI).
If `iw3-player-gui.bat` is missing, run `update.bat` followed by `update-installer.bat` to generate it.

![GUI](https://github.com/user-attachments/assets/bdf2666e-bd2a-4b3f-827b-fb95be174507)

### Command Line (CLI)

You can also start the server using the following commands:

**GUI Mode**
```bash
python -m iw3.player.gui
```

**CLI Mode**
```bash
python -m iw3.player --root /path/to/your/media
```

#### Common Options
- `--root PATH`: Path to media directory (default: `./example`)
- `--port PORT`: Port number (default: `1304`)
- `--bind-addr IP`: IP address (default: local IP address)
- `--user USER` / `--password PASS`: Username and password for authentication
- `--clear-cache`: Clear thumbnail cache on startup

Once the server is running, access the URL displayed in the terminal or GUI (e.g., `https://[IP address]:1304`) using the browser on your VR device.

#### About SSL Warnings

To meet WebXR's mandatory HTTPS requirement for local streaming, the iw3-player backend generates a self-signed certificate on startup. 

During your first visit, your browser will display a security warning (e.g., "Your connection is not private"). This is expected. To proceed:

1. Click **"Advanced"**.
2. Select **"Proceed to [IP address]"**.

**Tip:** Bookmarking this page will allow you to skip this warning in the future. 

While browsers typically discourage trusting local servers in this manner, it is a necessary workaround for local WebXR development.

## UI and Controls

### Basic VR Controller Operations

| Operation | VR Controller | Keyboard / Mouse |
| :--- | :--- | :--- |
| **Menu Toggle** | B / Y buttons, or short trigger click in empty space | Space |
| **Select / Confirm** | Trigger (Index finger) | Mouse click |
| **Navigate Images** | Stick Left/Right (during image playback) | Left / Right Arrow keys |
| **Seek (Video)** | Stick Left/Right (during video playback) | Left / Right Arrow keys |
| **Page Navigation** | Stick Up/Down (in File Explorer) | - |
| **Play / Pause** | A / X buttons (during video playback) | - |
| **Zoom (Distance)** | Stick Up/Down (when pointing at background) | Up / Down Arrow keys / Mouse wheel |
| **Disparity (Scale)** | `[Grip + Stick Y]` | Q / E keys |
| **Screen Size** | `[Grip + Stick X]` | - |
| **Tilt / Roll Adjustment**| `[Long-press Trigger + Controller Move]` (pointing at background) | - |
| **Pan / Position (Pos X / Y)** | `[Grip + Move Controller]` | Mouse drag |
| **Exit XR** | Power icon at the top-right of the Main Menu | - |

*Note: An alignment guide is displayed during VR180 dragging to help center the view and adjust tilt.*

#### Main Menu

A clean menu separating navigation and playback controls.

- **Navigation**: File Explorer, Open Recent File, Repeat playback mode toggle, Reset Settings, Exit XR
- **Status Display**: Real-time display of current filename, time, battery level, and rendering FPS
- **Playback Controls (Video)**:
  - **Seek Bar & Time**: Timeline scrubbing, Play/Pause
  - **Volume Controls**: Volume slider and Mute toggle button
  - **Playback Speed**: Adjust video playback speed from 0.25x to 4.00x using `[−]` / `[+]` buttons; click the speed icon to reset to 1.00x
  - *Note: Volume and playback speed are automatically remembered for future sessions.*

#### Screen Settings

This is the most frequently used menu, allowing you to fine-tune the screen's shape and its position in 3D space.  
(`Stick X` is thumbstick horizontal, `Stick Y` is thumbstick vertical, `Move` is physical controller movement)

| Item | Shortcut | Description |
| :--- | :--- | :--- |
| `Screen Size` | `[Grip + Stick X]` | Physical size of the virtual screen (1m to 16m) |
| `Tilt` | `[Long-press Trigger + Tilt]` | Vertical tilt of the screen (Roll/Tilt in VR180) |
| `Distance` | `[Stick Y]` | Distance from the viewer (Zoom) |
| `Scale` | `[Grip + Stick Y]` | Stereoscopic disparity / Eye Separation (Perceived object size) |
| `Pos X` / `Pos Y` | `[Grip + Move]` | Move screen horizontally or vertically |
| `Curvature` | - | Adjusts the horizontal curve of the screen |
| `Edge Fade` | - | Smoothly fades out the edges of the screen |
| `Background` | - | Adjusts background brightness (Ignored if a skybox is active) |

*Note: In VR180 mode, settings exclusive to flat screens are automatically disabled.*  
*Note: Adjustments are automatically saved per file. Unconfigured files inherit the previous screen settings.*

These operations can sometimes result in perceptual effects that differ from the literal mechanical change. We highly recommend reading the [Screen Operation and Perception](#screen-operation-and-perception) section for the best experience.

#### Color Settings

Apply real-time color correction and image enhancements. While most media looks best with default settings, these tools are invaluable for specific sources.

- **LUT (Color Profile)**: Select from various cinematic or technical color profiles.
- **Adjustments**: Fine-tune `Contrast`, `Brightness`, `Hue`, `Saturation`, and `Gamma`.

You can add custom LUTs by placing `.cube` files in the `public/lut/` directory.  
Using `haldclut2cube.py`, you can convert the effect applied to [Identity HaldCLUT image](https://rawpedia.rawtherapee.com/File:Hald_CLUT_Identity.tif) into a `.cube` file.

#### Env Settings (Environment)

Customize your virtual theater. While not strictly necessary for viewing, watching media under a high-fidelity sky can greatly improve the experience.

- **Skybox**: Set a 360-degree background image and adjust its orientation, tilt, and brightness.
- **3D Model**: Load a 3D scene model (e.g., a floor or room), with adjustable size and ground height.

Custom backgrounds (`.hdr` or `.exr`) and models (`.glb`) can be added to the `public/environments/` directory. High-quality HDRI and IBL assets are available at [Poly Haven](https://polyhaven.com/) and [ambientCG](https://ambientcg.com/). Scenes created in Blender can be exported as `.glb` (supporting Draco compression and animations) for use here.

#### Subtitle Settings

Configuration for video subtitles.

- Toggle subtitles ON/OFF.
- Switch between multiple subtitle tracks.
- Adjust **Vertical Position**, **Depth**, **Font Size**, and **Eye Separation** (disparity).

*See [Adjusting Subtitle Position](#adjusting-subtitle-position) for tips on comfortable 3D reading.*

#### Render Settings

Advanced settings for the rendering engine. **Changes require a browser reload to take effect.**

- **Font**: Select the UI font/language (primarily for the file explorer).
- **Antialias**: Smooths jagged object outlines (aliasing).
- **Super Sampling**: Increases resolution (pixel density) within VR.
- **Target Frame Rate**: Sets the target FPS for WebXR (e.g., 72, 90, 120Hz).
- **High Quality Video**: Plays VR180 videos using WebXR Layers (XRMediaBinding) to feed decoded video frames directly into the VR compositor for enhanced quality and lower GPU overhead (supported environments only; enabled by default).

These settings involve a trade-off between visual fidelity and battery/performance:

- **Antialias**: Can be disabled if you use `Edge Fade` to hide screen borders.
- **Super Sampling**: Setting this to 1x may result in poor image quality on Meta Quest devices due to their default resolution.
- **High Quality Video**: Can generally be left enabled; automatically falls back to standard rendering if unsupported by the browser.

##### About Fonts and Languages

The iw3-player UI uses Multi-channel Signed Distance Fields (MSDF) for crisp text rendering. Using an incorrect font may prevent non-Latin characters from appearing correctly.

| Font | Supported Content |
| :--- | :--- |
| `NotoSansJP` | Japanese (JIS Level 1 & 2) |
| `NotoSansSC` | Simplified Chinese (Standard table) |
| `NotoSansTC` | Traditional Chinese (MOE Standard) |
| `NotoSansKR` | Korean (All Hangul syllables) |
| `NotoSans` | Latin / Cyrillic (English, Russian, European, etc.) |

The player automatically selects a font based on your browser's language setting. If text appears garbled, please select a font manually.

*Note: Subtitles are rendered using standard browser Canvas features and are not affected by these font selections.*

## Security and Privacy

### Server Access

The backend server starts without a password by default.
It can only be accessed from within the same local network, but you can set a username and password if you feel it's necessary.

### Backend Storage

The backend saves thumbnails of images and videos, and subtitle data as a cache.
These cannot be seen in the OS explorer, etc., but raw data is saved.
There are several ways to delete it:

- Delete `nunif/tmp/iw3_player_cache`
- Start the backend with the `--clear-cache` option (deletes on every startup)
- Access `/factory_reset.html` on the server to perform a factory reset

The keys for thumbnails use a SHA256 hash value of the filename and the last modified date, and raw filenames are not saved.

### Browser Storage

Browser-side data is saved in IndexedDB.
This includes the following data:

- Global player settings (volume, playback speed, display preferences, etc.)
- Screen settings per file (disparity, screen size, tilt, stereo format, etc.)
- Path of the last opened file

While this cannot be read by any site other than the one running on the same address:port, care is taken not to save raw filenames.
In the per-file settings, the key is a SHA256 hash of the `filename + last modified date`. Raw filenames are not saved.
The path of the last opened file is encrypted with AES-GCM using a key generated by the backend at first startup. It can be decrypted if the same backend is present, but cannot be decrypted with browser-side data alone.

If you want to delete it, please access `/factory_reset.html` to perform a factory reset.

## Screen Operation and Perception

When viewing stereoscopic images, the physical adjustments made to the screen in 3D space do not always align with your psychological perception of the image. This section explains the perceptual effects of each control and recommends a workflow for the best viewing experience.

### Scale

The Scale setting adjusts the baseline between the left and right cameras (also known as IPD or Eye Separation). By shifting the horizontal offset, you can control the **perceived size** of objects in the scene. Use this to make people look like their actual height, or shrink them down for a "toy" effect.

### Screen Size and Distance

Both changing the physical screen size and changing its distance (Zoom) can be perceived as a "zoom" operation. However, iw3-player is designed to use **Distance** as the primary method for zooming. 

The **Screen Size** control should be used as a secondary adjustment when you reach the limits of other settings:

1. Use **Distance** for standard zooming.
2. Use **Screen Size** (Increase) if you want to get closer than the minimum distance.
3. Use **Screen Size** (Decrease) if you want to make objects smaller than the minimum scale allows.

### Perceived Depth and Volume

The "thickness" or 3D volume of objects is affected by several factors:

1. **Distance**: Objects appear flatter as the screen moves closer.
2. **Scale**: Objects appear flatter as the scale is decreased.
3. **Disparity**: Objects appear flatter if the original media has low 3D strength.

The 3D strength is specified during 3D conversion in **iw3**. On iw3-player, only the Distance and Scale can be adjusted.

### Recommended Workflow for Immersive Viewing

To make a scene feel "real," adjust your settings in this order:

1. If people look too small: **Increase Scale**.
2. If people look too large: **Decrease Scale**.
3. If people look too thin (flat): **Increase Distance**.
4. If the depth is comfortable: You can **Decrease Distance** for more impact.
5. If subjects are still too large at minimum scale: **Decrease Screen Size**.
6. If subjects are still too far at minimum distance: **Increase Screen Size**.

With practice, you can achieve perfect life-size immersion in seconds using just one hand.

## Adjusting Subtitle Position

Reading soft subtitles in 3D video can be challenging due to conflicting visual cues. This difficulty arises from the mixture of two types of disparity:

1. **Simulated disparity**: Created by the difference between the left and right media tracks (the source 3D effect + the `Scale` setting).
2. **Positional disparity**: Created by the physical placement of the subtitle panel in the 3D VR space.

Standard 3D subtitles only have positional disparity, while the video has both. 
To solve this, use the **Eye Separation** setting in the Subtitle menu to add simulated disparity to the text. Adjust the subtitles so they appear to float slightly in front of the video plane for the most comfortable reading experience.

## About the Development Environment

iw3-player was developed through the chat interface of [gemini-cli](https://github.com/google-gemini/gemini-cli).
More than 99% of the backend and frontend code was written by Gemini based on detailed specifications and my repeated feedback during testing.

The initial development spanned approximately two weeks with a cost of about $100 in Gemini API Key (Tier 1) usage.

### How to build vendor.bundle.js

iw3-player is developed with vanilla JavaScript, but because the dependencies for `@pmndrs/uikit` are complex, external libraries are bundled using npm.
Normally, pre-built files are downloaded automatically, but you can build them manually using the following commands:

```bash
npm install
npm run build
```

The output file will be generated in `public/lib`.

### Dependency on `nunif`

The core backend (CLI) and frontend do not strictly depend on the `nunif` framework. They can operate standalone by configuring `dir_config.py`. 

However, the GUI and certain other utility scripts are integrated with and dependent on `nunif`.
