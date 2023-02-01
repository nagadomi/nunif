# waifu2x Command Line Interface

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

4x + Denoise level 3
```
python -m waifu2x.cli -m noise_scale4x -n 3 -i tmp/images -o tmp/out
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

With TTA, increase mini-batch size
```
python -m waifu2x.cli --tta --batch-size 16 -m scale -i tmp/images -o tmp/out
```
