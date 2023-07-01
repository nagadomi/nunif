# iw3

I want to watch any video as 3D video on my VR device.
So I developed this very personal tool.

iw3 provides the ability to convert any 2D image/video to side-by-side 3D image/video. However, it does not support Anime.

This project is under construction.

## Overview

- Estimating depth image using [ZeoDepth](https://github.com/isl-org/ZoeDepth)
- Generating side-by-side image using grid_sample based lightweight model that is trained with [stable-diffusion-webui-depthmap-script](https://github.com/thygate/stable-diffusion-webui-depthmap-script) 's method

## Usage

Run `iw3` from the root directory of nunif.

```
python -m iw3 -i <input file or directory> -o <output file or directory>
```
The first time run takes time to download a large model file.

When you specify a directory with `-i` option, only the image files will be processed.

If you want to process list of videos, specify a `.txt` file containing a list of file paths. (UTF-8 encoding is expected)

```
python -m iw3 -i video_list.txt -o output_dir
```


Show options.
```
python -m iw3 -h
```

## About VR Player

I have tested the results with the following software.

### Pigasus VR Media Player

Pigasus works perfectly for SBS 3D image and video and Samba drive.
However, I am not a big fan of this user interface and controller.
If you can only choose one software, this is the one I recommend.

### SKYBOX VR Video Player

I like this user interface, but

- Loading image files from Samba drive is not supported (from the internal drive is supported)
- Aspect ratio is incorrect for SBS 3D video (See https://forum.skybox.xyz/d/407-full-sbs-3d , if the corect aspect ratio is a typical aspect ratio such as 4:3 or 16:9, you can manually fix it)

## About file naming rule

VR Player detects media format by filename.
Adding `_LRF` suffix to the filename will identify the file as full side-by-side 3D media.

When a directory is specified with `-o` option, it is automatically output as a filename with `{original_filename}_LRF.(png|mp4)`.

## Trouble shooting

### Very flat foreground

This tends to happen with outdoor scene photos. Maybe ZoeDepth's training data is not suitable for the purposes of this project.

Try
- `--remove-bg` option
- `--depth-model ZoeD_NK` option

### Video encoding error

Please post to the issue about the format of the video.

### 60fps video drops to 30fps

By default, it is limited to 30fps.
Use `--max-fps 128` option.

### It's a giant!

This is a problem with SBS 3D video that it cannot be rendered in actual size scale.

You can try adjusting it manually.

1. Use `--pad` option to adjust the frame scale
2. Adjust IPD scale 

On SKYBOX Player, set the 3D effect slider to around < -0.3.

On Pigasus, set `Settings > Advanced Settings > IPD` slider to large.

### Recommended workflow for slow processor

- Take some screen captures from the video
- Convert it to 3D photo and check the result on VR device
- If the results are acceptable, process the video

## Limitation

`--method row_flow`(by default) is currently only trained with `--divergence 2`(by default) And input width <= 1920px.
