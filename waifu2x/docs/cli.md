# waifu2x Command Line Interface

The following line executes the CLI command.
```
python -m waifu2x.cli -h
```
```
options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        model dir
  --noise-level {0,1,2,3}, -n {0,1,2,3}
                        noise level
  --method {scale4x,scale,noise,noise_scale,noise_scale4x,scale2x,noise_scale2x}, -m {scale4x,scale,noise,noise_scale,noise_scale4x,scale2x,noise_scale2x}
                        method
  --gpu GPU [GPU ...], -g GPU [GPU ...]
                        GPU device ids. -1 for CPU
  --batch-size BATCH_SIZE
                        minibatch_size
  --tile-size TILE_SIZE
                        tile size for tiled render
  --output OUTPUT, -o OUTPUT
                        output file or directory
  --input INPUT, -i INPUT
                        input file or directory. (*.txt, *.csv) for image list
  --tta                 use TTA mode
  --disable-amp         disable AMP for some special reason
  --image-lib {pil,wand}
                        image library to encode/decode images
  --depth DEPTH         bit-depth of output image. enabled only with `--image-lib wand`
  --format {png,webp,jpeg}, -f {png,webp,jpeg}
                        output image format
  --style {art,art_scan,scan,photo}   style for default model (art/art_scan/scan/photo). Ignored when --model-dir option is specified.
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
