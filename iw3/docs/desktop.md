# iw3 dekstop streaming

(Warning: This is a very experimental tool)

`iw3.desktop` is a tool that converts your PC desktop screen into 3D and streaming via WiFi.
It can be viewed as side-by-side 3D from the browser on Meta Quest.

You can watch any image and video displayed on your PC in realtime.

To control the PC, you will use the PC's keyboard and mouse. Audio is also used from the PC. Meta Quest is used only for the display/monitor.

Be careful of 3D sickness, as the depth estimation results for GUI windows and text are probably not good.
Basically, iw3.desktop is designed to be used for full-screen playback for images and videos.

## Security Notice

iw3.desktop starts an HTTP server with no password by default.

Note that other PCs may access the server if they are inside the same network.

You can specify the password for Basic Authentication with `--password` option.

## Launching the Server on PC

The following command launches the server.

```
python -m iw3.desktop
```
(If you are using nunif-windows-package on Windows, run `nunif-prompt.bat` and enter the command from the console.)

If the server is successfully launched, a following message will be shown.

```
Open http://192.168.11.6:1303
Estimated FPS = 30.24, Streaming FPS = 0.00
```
(`192.168.11.6` address depends on your network environment)

If a Firewall dialog appears, allow it.

Open the URL in your PC browser and check if the video can be played.
I have confirmed that the web page works with Google Chrome and Meta Quest 2 Browser. It does not work on Firefox.

If the IP address of the PC is not detected correctly, you can specify it with `--bind-addr` option.

```
python -m iw3.desktop --bind-addr 192.168.1.2
```

##  Viewing on Meta Quest

I have confirmed that it works with Meta Quest 2.

You can playback the video in 3D on Meta Quest by following the steps.

1. Run `Browser`
2. Enter the server URL
3. (Optional) Add URL to favorites
4. Play the video
5. Make **Browser** full screen using the icon in the top right corner of Browser.
6. Make the **video** full screen using the icon in the bottom right corner of the video.
7. Set to `Display Mode > 3D Side-by-Side` using the screen icon in the menu at the bottom of Browser
8. (Optional) Set to Curve Window

Note that `Display Mode` cannot be changed unless both the video and Browser are in full screen mode.

After that, you can use the PC keyboard and mouse to operate the displayed screen.

## Options

### Network

You can specify the address to launch the HTTP server with `--bind-addr` and `--port` options.

```
python -m iw3.desktop --port 7860
```

If you want to publish your server on the Internet (not recommended).
```
python -m iw3.desktop --bind-addr 0.0.0.0 --port 7860
```

### Authentication

You can configure HTTP Basic Authentication with `--user` and `--password` options.

```
python -m iw3.desktop --password iw3
```
```
python -m iw3.desktop --user admin --password 1234
```

### Resolution (video resolution)

You can specify the height of the screen with `--stream-height` option. 1080px by default.

```
python -m iw3.desktop --stream-height 720
```

### FPS

You can specify the streaming FPS with `--stream-fps` option. 15 FPS by default.

```
python -m iw3.desktop --stream-fps 30
```

If `Estimated FPS` is significantly lower than the specified FPS, the PC performance is not sufficient to process the specified FPS.

FPS will be much lower than the video conversion due to `--batch-size 1` processing.

Also, probably due to browser limitations, higher than `Streaming FPS = 30` is not achievable.

### Stereo setting

You can specify the same options as in GUI/CLI.

```
python -m iw3.desktop --depth-model ZoeD_Any_N --divergence 2 --convergence 0.5 --resolution 518
```

`--depth-model Any_V2_S --divergence 1 --convergence 1` by default.
