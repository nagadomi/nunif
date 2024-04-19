# iw3

I want to watch any 2D video as 3D video on my VR device, so I developed this very personal tool.

iw3 provides the ability to convert any 2D image/video into side-by-side 3D image/video. However, it does not support Anime.

This project is under construction.

[日本語の説明](docs/gui_ja.md)

## Overview

- Estimating depthmap using [ZeoDepth](https://github.com/isl-org/ZoeDepth) or [Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- Generating side-by-side image using grid_sample based lightweight model that is trained with [stable-diffusion-webui-depthmap-script](https://github.com/thygate/stable-diffusion-webui-depthmap-script) 's method

## Usage

First see the [installation section](../README.md#install).

### GUI

Run `iw3.gui` module from the root directory of nunif.

```
python -m iw3.gui
```

On Windows, run `Run iw3 GUI.bat`.

![iw3-gui](https://github.com/nagadomi/nunif/assets/287255/3dba4f23-395d-48eb-afb6-819c332a93ac)

### CLI

Run `iw3` or `iw3.cli` module from the root directory of nunif.

```
python -m iw3 -i <input file or directory> -o <output file or directory>
```

The following command shows all available options.
```
python -m iw3 -h
```

Also, when running `iw3` for the first time, it may take some time as it needs to download large model files.

### What is `--divergence`(`-d`) option?

(`3D Strength` in GUI)

![divergence](https://github.com/nagadomi/nunif/assets/287255/814e34ff-88bc-4d55-8f53-921a698bc8c5)

`2.0` by default. You can also specify `2.5`.

### What is `--convergence`(`-c`) option?

![convergence](https://github.com/nagadomi/nunif/assets/287255/ca8ce084-e02f-4098-8f4b-726bd6505f60)

- `0` is good, but screen edge areas are hard to see.
- `1` is the most friendly for curved display setting.
- `0.5` by default.

### What is `--ipd-offset` option?

(`Your Own Size` in GUI)

![ipd-offset](https://github.com/nagadomi/nunif/assets/287255/9ae7c504-08eb-4105-af36-b5a9da5b5ed8)

This may be adjustable on the VR Player. If so, set it to 0 (by default).

### What is `--foreground-scale` option?

When specifying a positive value (1 .. 3), foreground depth is scaled up and background depth is scaled down.

![foreground-scale](https://github.com/nagadomi/nunif/assets/287255/5664ea7a-bcf8-4430-b490-7f2bcf1a81c4)

To be used for outdoor photos where foreground(people) look very flat. For videos, `0`(by default) is recommended.


When specifying a nagative value (-1 .. -3), background depth is scaled up and foreground depth is scaled down.

![foreground-scale-negative](https://github.com/nagadomi/nunif/assets/287255/458ca299-fb23-4e8d-b530-c7a3fe17dee3)


Note that the transformation formula is different for ZoeDepth models(`ZoeD_N`, `ZoeD_Any_N`) and DepthAnything models(`Any_S`, `Any_B`, `Any_L`), even for the same value.


### What is `--edge-dilation` option?

(`Edge Fix` in GUI)

This parameter is used only for DepthAnything models (`Any_S`, `Any_B`, `ANY_L`).

DepthAnything model outputs very accurate depth, but in stereo generation, it causes artifacts at foreground and background edges.

This approach reduces artifacts by dilating foreground segments (high value area).

![edge-dilation](https://github.com/nagadomi/nunif/assets/287255/cb67b93a-bf26-4ea2-ac8b-418d5dc716c3)

`0` is disabled. `2` by default. `4` is the most eye-friendly, but it degrades depth accuracy.

## About VR Player

I have tested the results with the following software.

### Pigasus VR Media Player

Pigasus works perfectly for SBS 3D videos, images, and Samba drive(SMB).
However, I am not a big fan of its user interface and user experience.

If you can only choose one software, I would recommend this one.

### SKYBOX VR Video Player

With recent updates(v1.1.6), most features of 3D Full SBS are now working.
However, the following features have not yet been implemented.

- No ability to navigate prev/next images with joystick
- Screen height position is not adjustable

## About file naming rule

VR Player detects media format by filename.
Adding `_LRF_Full_SBS` suffix to the filename will identify the file as full side-by-side 3D media.

When specifying a directory with `-o` option, it is automatically output as a filename with `{original_filename}_LRF_Full_SBS.(png|mp4)`.

Reference:
- Pigasus requires `LRF` https://hanginghatstudios.com/pigasus-faq/#acc-tb_obg1300-0
- SKYBOX requires `Full_SBS`, https://forum.skybox.xyz/d/2161-skybox-vr-quest-v116-added-multi-language-keyboard-and-casting
- DeoVR requires `SBS` or `LR`(`LRF` seems to not work), https://deovr.com/app/doc#naming

I confirmed that `_LRF_Full_SBS` works with all of the above software.

## VR180 format

When `--vr180` option is specified, the video is output in VR180 format (equirectangular).

This is usually not recommended because of poor usability during playback.

This is useful if your video player does not have the ability to play Full SBS 3D videos or if you want to post the video on Youtube.

## Half SBS format

When `--half-sbs` option is specified, the video is output in Half SBS format (subsampled at half resolution).

Older VR devices may only support this format. Also, you may need to add `_3dh_` to the filename to play it.

## Anaglyph 3D format

When `--anaglyph` option is specified, the video is output in Red-Cyan Anaglyph 3D format.

(On GUI, select `Anaglyph *` option in `Stereo Format`)

Currently, the following methods are supported.

| Method    |                        |
|-----------|------------------------|
| color     | Color anaglyphs. Partial color reproduction. Retinal rivalry.
| half-color| Half-Color Anaglyphs. Partial color reproduction. Less retinal rivalry than color anaglyphs.
| gray      | Grayscale Anaglyphs. No color reproduction.
| wimmer    | Wimmer's Optimized Anaglyphs.
| wimmer2   | Wimmer's Improved Method.
| dubois    | Dubois Method.
| dubois2   | Dubois Method. This version ignores the specification `The first and second terms should be clipped to [0,1] before adding`.

Reference:
- [Anaglyph Methods Comparison](https://3dtv.at/Knowhow/AnaglyphComparison_en.aspx)
- [Conversion of a Stereo Pair to Anaglyph with the Least-Squares Projection Method Eric Dubois, March 2009](https://www.site.uottawa.ca/~edubois/anaglyph/LeastSquaresHowToPhotoshop.pdf)

For video, I recommend `--pix-fmt yuv444p` or `--pix-fmt rgb24` option. `yuv420p` (by default) uses 4:2:0 chroma subsampling, red colors are degraded and cause ghosting artifacts (crosstalk).

JPEG have the same problem so I recommend using PNG (by default) instead.

Also, `--convergence 0.5 --divergence 2.0` is recommended.

## Trouble shooting

### Very flat foreground

This tends to happen with outdoor scene photos.

There are several ways to fight this problem.

- Try `--foreground-scale 3` option
- Try`--remove-bg` option
- Try combined option `--divergence 4 --convergence 0 --foreground-scale 3 --remove-bg`

When `--remove-bg` is specified, the background area is removed using [rembg](https://github.com/danielgatis/rembg) with [U2-net](https://github.com/xuebinqin/U-2-Net)'s human segmentation model, before estimating depthmap.

### Video encoding error

Please post to the issue about the format of the video.

### 60fps video drops to 30fps

By default, FPS is limited to 30fps.
Use `--max-fps 128` option.

### It's a giant!

This is a problem with SBS 3D video that it cannot be rendered in actual size scale.

You can try adjusting scale manually.

- Adjust IPD offset on VR Player
- Use `--ipd-offset` option(`You own size` in GUI) to adjust IPD offset

It is better to adjust IPD offset on VR Player, but you can also apply IPD offset to the output image.

On SKYBOX Player, set the 3D effect slider to around < -0.3.

On Pigasus, set `Settings > Advanced Settings > IPD` slider to large.

Also, on Pigasus, you can zoom and pan the image by double-clicking the trigger button on the VR controller.

### CUDA Out of Memory

Use `--low-vram` option.

I tested this program on RTX 3070 Ti (8GB VRAM, Linux) and GTX 1050 Ti (4GB VRAM, Laptop, Windows).
Both work with the default option.

### How to convert rotated(height width swapped) video correctly

Use `--rotate-left`(rotate 90 degrees to counterclockwise) or `--rotate-right`(rotate 90 degrees to clockwise) option to fix the rotation.

### Video is interlaced

Use `--vf yadif` option to deinterlace the video source.
You can use ffmpeg's video filter with `--vf` option.

Note that
- only serial pipeline is supported

See https://ffmpeg.org/ffmpeg-filters.html

### Recommended workflow for slow processor or very large video

First check the results with a few samples. There are two ways.

The following command processes video keyframes about every 4 seconds and outputs the result as 3D photos (image files).
```
python -m iw3 --keyframe --keyframe-interval 4 -i input_video.mp4 -o output_dir/
```

The following command processes video frames every 2 seconds and outputs the result as 3D slideshow-like video (video file).
```
python -m iw3 --max-fps 0.5 -i input_video.mp4 -o output_dir/
```

If the results are acceptable, process the full video.

## Limitation

`--method row_flow`(by default) is currently only trained for the range `0.0 <= divergence <= 2.5` and `0.0 <= convergence <= 1.0`.

## About row_flow model and its training

See https://github.com/nagadomi/nunif/issues/60 .

Basically, fine tuning for this model is not necessary.
Perhaps what is needed is fine tuning for ZoeDepth.

## Monocular Depth Estimation Models

| Short Name  |                   |
|-------------|-------------------|
| `ZoeD_N`    | ZoeDepth model NYUv2. Tuned for indoor scenes.
| `ZoeD_K`    | ZoeDepth model KITTI. Tuned for outdoor scenes (dashboard camera view).
| `ZoeD_NK`   | ZoeDepth model NYUv2 and KITTI.
| `ZoeD_Any_N`| DepthAnything metric depth model NYUv2. Tuned for indoor scenes.
| `ZoeD_Any_K`| DepthAnything metric depth model KITTI. Tuned for outdoor scenes (dashboard camera view).
| `Any_S`     | DepthAnything model small. The most efficient model, with promising performance.
| `Any_B`     | DepthAnything model base.
| `Any_L`     | DepthAnything model large. This model gives high quality, but is also heavy in terms of computation.

Personally, I recommend `ZoeD_N`, `Any_B` or `ZoeD_Any_N`.
`ZoeD_Any_N` looks the best for 3D scene. `Any_B` has the most accurate foreground and background segmentation, but the foreground looks slightly flat.


## Stereo Generation Method (Left-Right Image Generation)

| Short Name  |                   |
|-------------|-------------------|
| `row_flow_v3_sym`| Calculating the backward warping(`grid_sample`) parameters with ML model. The left and right parameters are fully symmetric. Faster than `row_flow_v3`, with less artifacts. Trained with `0.0 <= divergence <= 5.0`. Default method.
| `row_flow_v3`    | Calculating the backward warping parameters with ML model. The left and right parameters are calculated individually. The output is closest to `apply_stereo_divergence_polylines` of `stable-diffusion-webui-depthmap-script`. Trained with `0.0 <= divergence <= 5.0`
| `row_flow_v2`    | Older version of `row_flow_v3`. Trained with `0.0 <= divergence <= 2.5`
| `forward_fill`   | Depth order forward warping. Works only for high-resolution images (Not recommended for small images or videos. May cause disparity banding artifacts). Very experimental method.
| `forward`        | `forward_fill` without hole fill. Just for debug.
| `grid_sample`,`backward`  | Naive backward warping. Lots of ghost artifacts. Just for debug.


## Updating torch.hub modules (ZoeDepth, DepthAnything)

When you manually update the code, run the following command.
```
python -m iw3.download_models
```

The command syncs the following repositories.

- https://github.com/nagadomi/ZoeDepth_iw3
- https://github.com/nagadomi/MiDaS_iw3
- https://github.com/nagadomi/Depth-Anything_iw3

If you already downloaded the model files (checkpoint filess), downloading model files will be skipped.
