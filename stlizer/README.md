# stlizer

stlizer is a fast conservative video stabilizer.

It basically just translates or rotates the video frames. It means that lens distortion or motion blur cannot be fixed, but there is no need to worry about unwanted distortions or artifacts caused by algorithms or AI models.

I use this software just for watching videos or for pre-processing for other software, so this is the best specification.

Synthetic video example
<div>
  <video width="640" height="320" controls="controls" preload="none">
    <source src="https://github.com/user-attachments/assets/06f6a6d2-8ab7-4d8b-bc07-ac39e676a974" type="video/mp4" />
  </video>
</div>

[Realworld video example from DeepStab dataset](https://github.com/user-attachments/assets/7ce12bfd-35f1-469e-aa61-72109d222c8e).

## Usage

```
python3 -m stlizer -i input_video.mp4 -o output_video.mp4
```

Show all options
```
python3 -m stlizer -h
```
Only important options are described below.

## --border option

You can specify out-of-frame processing with `--border` option.

### `--border black`

Simply fill with black color. The output resolution will be the same as the input video.

<div>
  <video width="320" height="320" controls="controls" preload="none">
    <source src="https://github.com/user-attachments/assets/4573e9c4-3d70-43a8-a476-d25be9c7e478" type="video/mp4" />
  </video>
</div>

### `--border crop`

Crop with the fixed ratio specified by `--padding` option (default 0.05 = -5% border cut).

The output resolution will be smaller then the input video.

<div>
  <video width="288" height="288" controls="controls" preload="none">
    <source src="https://github.com/user-attachments/assets/d74f8421-5b7f-4ae8-b0b5-9248e9eb528d" type="video/mp4" />
  </video>
</div>

### `--border expand`

Simply fill with black color. Add border padding secified by `--padding`(default 0.05 = 5% addtional border).

The output resolution will be larger then the input video.
There is less potential for some pixels of the frame to be cut off.

<div>
  <video width="352" height="352" controls="controls" preload="none">
    <source src="https://github.com/user-attachments/assets/f5a7b93c-7bf3-4434-857d-e8a35805755b" type="video/mp4" />
  </video>
</div>

### `--border outpaint`

Fill with coarse blurry outpainting.

<div>
  <video width="320" height="320" controls="controls" preload="none">
    <source src="https://github.com/user-attachments/assets/95585d66-1e8a-42cf-84c6-d9a6b30e09cd" type="video/mp4" />
  </video>
</div>

This would be suitable for input to other software such as iw3. However, I have not tested it yet.

### `--border expand_outpaint`

outpaint version of `expand`.

<div>
  <video width="352" height="352" controls="controls" preload="none">
    <source src="https://github.com/user-attachments/assets/cd889b5b-fc0d-4862-8143-74f4b53a77fb" type="video/mp4" />
  </video>
</div>

## Smoothing Strength

You can specify the smoothing strength with `--smoothing` option (2 by default).

If a higher value is specified, the original camera trajectory will not be respected and the out-of-frame area will increase.

## About Video Analysis Cache (`*.stlizer`)

`*.stilizer` file is a cache file for pre-computed camera trajectory. It does not include raw file names or any privacy-sensitive data.

If this file exists, pass1 and pass2 can be skipped, when the options are compatible.

The storage location depends on OS.
```
Linux: ~/.cache/stlizer
macOS: ~/Library/Caches/stlizer
Windows: %APPDATA%\\Local\\nunif\\stlizer\\Cache
```

When `--disable-cache` option is specified, the cache file is ignored and not created.

All cache files can be deleted with the following command.
```
pytyon3 -m stlizer.purge_cache
```
