# iw3

I want to watch any 2D video as 3D video on my VR device, so I developed this very personal tool.

iw3 provides the ability to convert any 2D image/video into side-by-side 3D image/video.

This project is under construction.

[日本語の説明](docs/gui_ja.md)

## Overview

- Estimating depthmap using [ZeoDepth](https://github.com/isl-org/ZoeDepth) or [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) or [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) or [Depth Pro](https://github.com/apple/ml-depth-pro) or [Distill Any Depth](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth) or [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything).
- Generating side-by-side image using grid_sample based lightweight model

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

`2.0` by default.

This parameter changes the best distance for the screen position.

With lower value, the best distance for the screen position is farther away.
With higher value, the best distance for the screen position is closer.
However, with higher value, artifacts are more visible.

You can adjust the screen position by zoom-in/zoom-out on the VR Video Player.


### What is `--convergence`(`-c`) option?

![convergence](https://github.com/nagadomi/nunif/assets/287255/ca8ce084-e02f-4098-8f4b-726bd6505f60)

- `0` is good, but screen edge areas are hard to see.
- `1` is the most friendly for curved display setting.
- `0.5` by default.

### What is `--ipd-offset` option?

(`Your Own Size` in GUI)

![ipd-offset](https://github.com/nagadomi/nunif/assets/287255/9ae7c504-08eb-4105-af36-b5a9da5b5ed8)

This may be adjustable on the VR Player. If so, set it to 0 (by default).

### What is `--synthetic-view` option?

With `both`, generate views for both eyes. With `left` or `right`, only one view is generated. The other side view will be the original image/frame.

When `both` is specified, artifacts/distortions are balanced across the left and right eyes. This reduces the artifacts, but may cause artifacts seen in both eyes.

When `left` or `right` is specified, one eye view will have no artifact/distortion because it is the original image, but the opposite eye view will have twice as much artifact/distortion. Whether `left` or `right` is better depends on your dominant eye.

At the moment, I recommend `both`. `both` by default.

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

## About Colorspace

See [Colorspace](docs/colorspace.md) and https://github.com/nagadomi/nunif/issues/164 .

## About Depth Resolution

See https://github.com/nagadomi/nunif/discussions/168 .

## About Video Format

`mp4` is a highly compatible format.

`mkv` can be previewed during conversion.

`avi` is provided to output lossless video for your own encoding.

## Video Codec

`libx264` is for H.264 which is a highly compatible format. However, at higher resolutions like 4K, the file size will be larger, which may cause playback lag/artifact.

`libx265` is for H.265.

`utvideo` is for lossless video. You may need [Ut Video Codec Suite](https://github.com/umezawatakeshi/utvideo/releases) for playback.

#### Level option for H.265

`auto` is recommended.
This will cause an error if the wrong level is selected for a video that is out of range for the specification.

#### H.264 Profile

There is no way to specify H.264 profile.
However, it seems to be `Constrained Baseline` when `--preset ultrafast` and `High` otherwise.

## About VR Player

I have tested the results with the following software.

### Pigasus VR Media Player

Pigasus works perfectly for SBS 3D videos, images, and Samba drive(SMB).
However, I am not a big fan of its user interface and user experience.

### SKYBOX VR Video Player

With recent updates, most features of 3D Full SBS are now working.

To adjust the screen position, you must select `Cinema Scene > SELECT THEATER > VOID`.

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

This is useful if your video player does not have the ability to play Full SBS 3D videos or if you want to post the video on Youtube (See https://github.com/nagadomi/nunif/issues/268 ) .

## Half SBS format

When `--half-sbs` option is specified, the video is output in Half SBS format (subsampled at half resolution).

Older VR devices may only support this format. Also, you may need to add `_3dh_` to the filename to play it.

## Full TB and Half TB format

When `--tb` or `--half-tb` option is specified, the video is output in TopBottom format.

TopBottom format can be played back with higher resolution than SBS on some 3D TVs (Polarized/Passive 3D system).

## Cross Eyed

When `--cross-eyed` option is specified, the video/image is output for cross-eyed viewing method.

Unlike the normal SBS format, the images are reversed left and right.

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

## Export and Export Disparity

Export and Export Disparity are features to output depth and frame images.

See https://github.com/nagadomi/nunif/issues/97#issuecomment-2027349722 for details at the moment.

# `Flicker Reduction` (`--ema-normalize`)

This feature stabilizes temporal variations in the depth range using exponential moving average.

There are two parameters for this.

1. Decay Rate (`--ema-decay`)
2. Lookahead Buffer Size (`--ema-buffer`)

`Decay Rate` accepts a value between 0 and 1. The larger the value, the smaller the change between frames. However, if it is too large, clipping may occur when the depth range changes suddenly. The appropriate range is between 0.75 and 0.99.

`Lookahead Buffer Size` specifies the number of future frames, starting from the current frame, over which the depth range is calculated.
Buffer Size = 30 means looking ahead 1 seconds at 30 FPS. Buffer Size = 150 means looking ahead 5 seconds. Buffer Size = 1 is the same as in the previous version.

## Scene Boundary Detection

Scene/Shot boundary detection is performed first, using [TransNetV2](https://github.com/soCzech/TransNetV2).

At scene boundary frames, state reset is performed if needed.

This includes:
- The state of VideoDepthAnything
- The state of Flicker Reduction (Min-Max Normalization)

## `Depth Anti-aliasing`

See https://github.com/nagadomi/nunif/issues/406

## Trouble shooting

### Output video is not SBS

Some software, such as Windows Photo, shows only one side of side-by-side layout.
Try playing it with other video players.

### Very flat foreground

This tends to happen with outdoor scene photos.

There are several ways to fight this problem.

- Try `--foreground-scale 3` option
- Try combined option `--divergence 4 --convergence 0 --foreground-scale 3`

### Video encoding error

Please post to the issue about the format of the video.

### Large video file size

You can reduce the file size with `--preset medium` option.
`--video-codec libx265` also helps reduce the file size.

### 60fps video drops to 30fps

By default, FPS is limited to 30fps.
Use `--max-fps 128` option.

Note that 60fps video takes twice as long to process as 30fps limit.

### Problems with older GPUs

`FP16` can cause slowdowns and errors on older GPUs (older than GeForce 20 series).

In CLI, `FP16` can be disabled with `--disable-amp` option.

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

### NVENC(`h264_nvenc`, `hevc_nvenc`) does not work

Install NVIDIA Driver 570 or newer.

It's also important to note that hardware encoders have resolution limitations, and exceeding those limits can cause encoding to fail with an error.
This issue is particularly common with VR90 (--vr180), as it produces very large video resolutions that are more likely to exceed hardware encoder limits.

### How to convert rotated(height width swapped) video correctly

Use `--rotate-left`(rotate 90 degrees to counterclockwise) or `--rotate-right`(rotate 90 degrees to clockwise) option to fix the rotation.

### Video is interlaced

Use `--vf yadif` option to deinterlace the video source.
You can use ffmpeg's video filter with `--vf` option.

Note that
- only serial pipeline is supported

See https://ffmpeg.org/ffmpeg-filters.html

### Artifacts in encoded video

When `preset=slower|veryslow|placebo` is used for high resolution video, it may produce video files that are not supported by hardware decoder.
If you have been using that setting, try `preset=medium`.

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

### "HIP error: invalid device function" on older AMD GPUs

Append the `HSA_OVERRIDE_GFX_VERSION` environment variable to your command. Below are examples launching the GUI module with different AMD GPUs.

For 6700, 6600 and other RDNA2 or older,
```
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m iw3.gui
```
or for AMD 7600 and other RDNA3 cards:
```
HSA_OVERRIDE_GFX_VERSION=11.0.0 python -m iw3.gui
```

## Limitation

`--method row_flow_v3`(by default) is currently only trained for the range `0.0 <= divergence <= 5.0` and `0.0 <= convergence <= 1.0`.

`--method mlbw_l2` and `--method mlbw_l4` are trained for the range `0.0 <= divergence <= 10.0` and `0.0 <= convergence <= 1.0`.

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
| `ZoeD_Any_N`| Depth-Anything Metric Depth model NYUv2. Tuned for indoor scenes.
| `ZoeD_Any_K`| Depth-Anything Metric Depth model KITTI. Tuned for outdoor scenes (dashboard camera view).
| `Any_S`     | Depth-Anything model small. The most efficient model, with promising performance.
| `Any_B`     | Depth-Anything model base.
| `Any_L`     | Depth-Anything model large. This model gives high quality, but is also heavy in terms of computation.
| `Any_V2_S`  | Depth-Anything-V2 model small.
| `Any_V2_B`  | Depth-Anything-V2 model base. (cc-by-nc-4.0)
| `Any_V2_L`  | Depth-Anything-V2 model large. (cc-by-nc-4.0)
| `Any_V2_N_S`| Depth-Anything-V2 Metric Depth model Hypersim small. Tuned for indoor scenes.
| `Any_V2_N_B`| Depth-Anything-V2 Metric Depth model Hypersim base. Tuned for indoor scenes
| `Any_V2_N_L`| Depth-Anything-V2 Metric Depth model Hypersim large. Tuned for indoor scenes. (cc-by-nc-4.0)
| `Any_V2_K_S`| Depth-Anything-V2 Metric Depth model VKITTI small. Tuned for outdoor scenes (dashboard camera view).
| `Any_V2_K_B`| Depth-Anything-V2 Metric Depth model VKITTI base. Tuned for outdoor scenes (dashboard camera view).
| `Any_V2_K_L`| Depth-Anything-V2 Metric Depth model VKITTI large. Tuned for outdoor scenes (dashboard camera view). (cc-by-nc-4.0)
| `DepthPro`  | Depth Pro model. 1536x1536 resolution. For image use.
| `DepthPro_S`  | Depth Pro model. 1024x1024 modified resolution. For image use.
| `Distill_Any_S`  | Distill Any Depth model small.
| `Distill_Any_B`  | Distill Any Depth model base.
| `Distill_Any_L`  | Distill Any Depth model large.
| `VDA_S`  | Video Depth Anything small.
| `VDA_L`  | Video Depth Anything large.
| `VDA_Metric`  | Video Depth Anything metric depth model.

Personally, I recommend `ZoeD_Any_N`, `Any_B` or `VDA_Metric`.
`ZoeD_Any_N` looks the best for 3D scene. The DepthAnything models have more accurate foreground and background segmentation, but the foreground looks slightly flat.

For art/anime, DepthAnything is better than ZoeDepth.

Regarding Depth Pro, distance is currently clipped at 40m. It is planned to become adjustable as an option.

### About `Any_V2_B` ,`Any_V2_L`, `Any_V2_N_L`, `Any_V2_K_L`

These models are licensed under cc-by-nc-4.0 (Non Commercial). These are not available by default due to conflicts with nunif MIT license.

If you want to use it, agree to the pre-trained model license and place the checkpoint file yourself.

| Short Name | Path |
|------------|------|
| `Any_V2_B` | `iw3/pretrained_models/hub/checkpoints/depth_anything_v2_vitb.pth`
| `Any_V2_L` | `iw3/pretrained_models/hub/checkpoints/depth_anything_v2_vitl.pth`
| `Any_V2_N_L` | `iw3/pretrained_models/hub/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth`
| `Any_V2_K_L` | `iw3/pretrained_models/hub/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth`

These files can be downloaded from Models section of https://huggingface.co/depth-anything .

- https://huggingface.co/depth-anything/Depth-Anything-V2-Base
- https://huggingface.co/depth-anything/Depth-Anything-V2-Large
- https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
- https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large


### About Video-Depth-Anything

#### `VDA_L`, `VDA_Metric`

These models are licensed under cc-by-nc-4.0 (Non Commercial).
If you want to use it, agree to the pre-trained model license and place the checkpoint file yourself.

| Short Name | Path |
|------------|------|
| `VDA_L` | `iw3/pretrained_models/hub/checkpoints/video_depth_anything_vitl.pth`
| `VDA_Metric` | `iw3/pretrained_models/hub/checkpoints/metric_video_depth_anything_vitl.pth`

These files can be downloaded from Models section of https://huggingface.co/depth-anything .

- https://huggingface.co/depth-anything/Video-Depth-Anything-Large
- https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large

#### VDA Implementation Notes

The following options are highly recommended.

- `Scene Boundary Detection` (`--scene-detect`)
- `Flicker Reduction` (`--ema-normalize`)

The following options are ignored.

- `Low VRAM` (`--low-vram`)
- `TTA` (`--tta`)
- `Worker Threads` (`--max-workers`)
- `Stream` (`--cuda-stream`)

`Batch Size`(`--batch-size`) is also ignored when using the depth model (32 is used), but it is used during preprocessing and stereo generation.

Also, the original implementation uses global min/max values for normalization, but iw3's online processing uses their moving average.

### About `Distill_Any_B`, `Distill_Any_L`

These models are stated to be under Apache License 2.0, but they use Depth-Anything-V2, which is licensed under cc-by-nc-4.0 (Non Commercial), as the initial weights.

If you want to use it, place the checkpoint file yourself.

| Short Name | Path |
|------------|------|
| `Distill_Any_B` | `iw3/pretrained_models/hub/checkpoints/distill_any_depth_vitb.safetensors`
| `Distill_Any_L` | `iw3/pretrained_models/hub/checkpoints/distill_any_depth_vitl.safetensors`

These files can be downloaded from Pre-trained Models section of https://github.com/Westlake-AGI-Lab/Distill-Any-Depth .

These files are in `.safetensors` format, so conversion to `.pth` is not required. Renaming to an appropriate file name is required.

## Stereo Generation Method (Left-Right Image Generation)

| Short Name  |                   |
|-------------|-------------------|
| `row_flow_v3`    | Calculating the backward warping parameters with ML model. Trained with `0.0 <= divergence <= 5.0` and synthetic training data generated by `forward_fill`. Default method.
| `mlbw_l2`    | Calculating the 2-layer backward warping parameters with ML model. Trained with `0.0 <= divergence <= 10.0`.
| `mlbw_l4`    | Calculating the 4-layer backward warping parameters with ML model. Trained with `0.0 <= divergence <= 10.0`.
| `mlbw_l2s`   | The small model of `mlbw_l2`. Trained with `0.0 <= divergence <= 5.0`. When `4.0 < divergence`, the same model as `mlbw_l2` is used.
| `mlbw_l4s`   | The small model of `mlbw_l4`. Trained with `0.0 <= divergence <= 5.0`. When `4.0 < divergence`, the same model as `mlbw_l4` is used.
| `row_flow_v2`    | Previous default model. Trained with `0.0 <= divergence <= 2.5` and synthetic training data generated by [stable-diffusion-webui-depthmap-script](https://github.com/thygate/stable-diffusion-webui-depthmap-script) 's method.
| `forward_fill`   | Depth order bilinear forward warping. Non ML method.
| `row_flow_v3_sym`| Calculating the backward warping(`grid_sample`) parameters with ML model. The left and right parameters are fully symmetric. 2x faster than `row_flow_v3`. For experimental use.
| `forward`        | `forward_fill` without hole fill. For experimental use.
| `grid_sample`,`backward`  | Naive backward warping. Lots of ghost artifacts. For experimental use.

## Multi-GPU

Multi-GPU(`All CUDA Device`) feature is supported only for video processing and Export. Other features work on a single GPU.

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

## Sub Project

[iw3-desktop](docs/desktop.md) is a tool that converts your PC desktop screen into 3D in realtime and streaming over WiFi.
