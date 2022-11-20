# Waifu2x

Download pre-trained models.
```
python -m waifu2x.download_models
```
This command downloads the pre-trained models to `waifu2x/pretrained_models`.

# Current limitations

none?

# CLI

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

Generate `waifu2x/public_html`.
```
python -m waifu2x.webgen.gen
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

## Use ReCAPTCHA

Copy `waifu2x/web-config.ini.sample` to `waifu2x/web-config.ini`
```
cp waifu2x/web-config.ini.sample waifu2x/web-config.ini
```

Edit `site_key` and `secret_key` in `web-config.ini`.

Run
```
python -m waifu2x.web --enable-recaptcha --config waifu2x/web-config.ini
```
