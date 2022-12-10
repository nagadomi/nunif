# Waifu2x

Download pre-trained models.
```
python -m waifu2x.download_models
```
This command downloads the pre-trained models to `waifu2x/pretrained_models`.

# Current limitations

Training is not supported yet.

The CLI and Web App allow you to switch the image library to be used with `--image-lib (pil|wand)`. When `--image-lib pil`(default), 16bit image output is not supported. If you want to use `--depth 16`, specify `--image-lib wand`.

When `--format jpeg` is specified, the transparent areas of the image will be set to white background and the alpha channel will be removed.

When `--format jpeg` or `--format webp` is specified, image-gamma value(gAMA) is ignored (removed from the output image).

# Note

In Ubuntu 22.04, gThumb(Image Viewer) has a bug in rendering webp's alpha channel. If you encounter any problems with the alpha channel, please check it with Google-chrome.

# Command Line Interface

The following line executes the CLI command.
```
python -m waifu2x.cli -h
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

With TTA, half/16bit float, mini-batch
```
python -m waifu2x.cli --tta --amp --batch-size 16 -m scale -i tmp/images -o tmp/out
```

# Web Application

Generate `waifu2x/web/public_html`.
```
python -m waifu2x.web.webgen.gen
```

The following line starts the Web Server.
```
python -m waifu2x.web
```
The web server starts at http://localhost:8812/ .

Show help.
```
python -m waifu2x.web -h
```

With TTA, half float, debug log print
```
python -m waifu2x.web --tta --amp --debug
```
or
```
DEBUG=1 python -m waifu2x.web --tta --amp --debug
```

# Image Encode/Decode

## Use reCAPTCHA

Copy `waifu2x/web/config.ini.sample` to `waifu2x/web/config.ini`
```
cp waifu2x/web/config.ini.sample waifu2x/web/config.ini
```

Edit `site_key` and `secret_key` in `waifu2x/web/config.ini`.

Run
```
python -m waifu2x.web --enable-recaptcha --config waifu2x/web/config.ini
```
