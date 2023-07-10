# waifu2x Command Line Interface

The following line executes the CLI command.
```
python -m waifu2x.cli -h
```
```
options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        model dir (default: None)
  --noise-level {0,1,2,3}, -n {0,1,2,3}
                        noise level (default: 0)
  --method {scale4x,scale2x,noise_scale4x,noise_scale2x,scale,noise,noise_scale}, -m {scale4x,scale2x,noise_scale4x,noise_scale2x,scale,noise,noise_scale}
                        method (default: noise_scale)
  --gpu GPU [GPU ...], -g GPU [GPU ...]
                        GPU device ids. -1 for CPU (default: [0])
  --batch-size BATCH_SIZE
                        minibatch_size (default: 4)
  --tile-size TILE_SIZE
                        tile size for tiled render (default: 256)
  --output OUTPUT, -o OUTPUT
                        output file or directory (default: None)
  --input INPUT, -i INPUT
                        input file or directory. (*.txt, *.csv) for image list (default: None)
  --tta                 use TTA mode (default: False)
  --disable-amp         disable AMP for some special reason (default: False)
  --image-lib {pil,wand}
                        image library to encode/decode images (default: pil)
  --depth DEPTH         bit-depth of output image. enabled only with `--image-lib wand` (default: None)
  --format {png,webp,jpeg}, -f {png,webp,jpeg}
                        output image format (default: png)
  --style {art,photo,scan,art_scan}
                        style for default model (art/scan/photo). Ignored when --model-dir option is specified. (default: None)
  --grayscale           Convert to grayscale format (default: False)
  --recursive, -r       process all subdirectories (default: False)
  --resume              skip processing when output file is already exist (default: False)
  --max-fps MAX_FPS     max framerate. output fps = min(fps, --max-fps) (video only) (default: 128)
  --crf CRF             constant quality value. smaller value is higher quality (video only) (default: 20)
  --preset {ultrafast,superfast,veryfast,faster,fast,medium,slow,slower,veryslow,placebo}
                        encoder preset option (video only) (default: ultrafast)
  --tune {film,animation,grain,stillimage,fastdecode,zerolatency} [{film,animation,grain,stillimage,fastdecode,zerolatency} ...]
                        encoder tunings option (video only) (default: ['zerolatency'])
  --yes, -y             overwrite output files (video only) (default: False)
  --rotate-left         Rotate 90 degrees to the left(counterclockwise) (video only) (default: False)
  --rotate-right        Rotate 90 degrees to the right(clockwise) (video only) (default: False)
  --vf VF               video filter options for ffmpeg.  (video only) (default: )
  --grain               add noise after denosing (video only) (default: False)
  --grain-strength GRAIN_STRENGTH
                        noise strength (video only) (default: 0.05)
  --pix-fmt {yuv420p,yuv444p}
                        pixel format (video only) (default: yuv420p)
```

When `DEBUG` environment variable is defined, DEBUG level log will be printed.
```
DEBUG=1 python -m waifu2x.cli -h
```

## CLI Examples

Denoise level 0 (-n noise_level)
```
python -m waifu2x.cli -m noise -n 0 -i tmp/images -o tmp/out
```


2x
```
python -m waifu2x.cli -m scale  -i tmp/images -o tmp/out
```

2x, webp output
```
python -m waifu2x.cli -m scale  -i tmp/images -o tmp/out -f webp
python -m waifu2x.cli -m scale  -i tmp/images/image.jpg -o tmp/out/image.webp
```

2x + Denoise level 3
```
python -m waifu2x.cli -m noise_scale -n 3 -i tmp/images -o tmp/out
```

4x + Denoise level 3
```
python -m waifu2x.cli -m noise_scale4x -n 3 -i tmp/images -o tmp/out
```

Using photo model (--style photo)
```
python -m waifu2x.cli --style photo -m noise_scale4x -n 3 -i tmp/images -o tmp/out
```

Also, for photo models, larger `--tile-size` will give better results (less tile seam/border artifact)
```
python -m waifu2x.cli --style photo -m noise_scale4x -n 3 --tile-size 640 --batch-size 1 -i tmp/images -o tmp/out
```

Using `art_scan` model (`--style scan` or `--style art_scan`)
```
python -m waifu2x.cli --style scan -m noise_scale4x -n 3 -i tmp/images -o tmp/out
```


With model dir
```
python -m waifu2x.cli --model-dir ./waifu2x/pretrained_models/upconv_7/photo/ -m noise_scale -n 1 -i tmp/images -o tmp/out
```

With multi GPU (--gpu gpu_ids)
```
python -m waifu2x.cli --gpu 0 1 2 3 -m scale -i tmp/images -o tmp/out
```

With TTA
```
python -m waifu2x.cli --tta -m scale -i tmp/images -o tmp/out
```
Note that TTA is not valid for `photo` and `art_scan`, which are GAN-based models.

With TTA, increase mini-batch size
```
python -m waifu2x.cli --tta --batch-size 16 -m scale -i tmp/images -o tmp/out
```

### video processing

First of all, the waifu2x models are designed for images, not for video stream.

examples
```
python -m waifu2x.cli --style art --method noise_scale -i tmp/input_video.mp4 -o tmp/output_dir
```
```
python -m waifu2x.cli --style art_scan --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

high quality encoding
```
python -m waifu2x.cli --crf 16 --preset medium --pix-fmt yuv444p --style art_scan --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

drop to 30fps from 60fps video (fps = min(original fps, `--max-fps`).
```
python -m waifu2x.cli --max-fps 30 --style art_scan --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

1fps video (for preview)
```
python -m waifu2x.cli --max-fps 1 --style art_scan --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

fix rotation (width height swap, `--rotate-left`(counterclockwise) or `--rotate-right`(clockwise))
```
python -m waifu2x.cli --rotate-left --style photo --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

deinterlace input video stream. (you can use ffmpeg's video filter with `--vf` option)
```
python -m waifu2x.cli --vf yadif --style photo --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

add noise after denosing
```
python -m waifu2x.cli --grain --style photo --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

use lightweight(older) models

`cunet/art` is best trade-off between performance and quality
```
python -m waifu2x.cli --model-dir ./waifu2x/pretrained_models/cunet/art --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```

`upconv_7/art` and `upconv_7/photo` are the fastest, but lowest quality
```
python -m waifu2x.cli --model-dir ./waifu2x/pretrained_models/upconv_7/art --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```
```
python -m waifu2x.cli --model-dir ./waifu2x/pretrained_models/upconv_7/photo --method noise -n 3 -i tmp/input_video.mp4 -o tmp/output_video.mp4
```
