# Waifu2x

Download pre-trained models.
```
python -m waifu2x.download_models
```
This command downloads the pre-trained models to `waifu2x/pretrained_models`.

Run CLI command from nunif root directory.
```
python -m waifu2x.cli -h
```

Run Web Service from nunif root directory (Experimental).
```
python -m waifu2x.web -h
```

# Current limitations

16bit RGB/RGBA is not suupported due to Pillow does not support for some 16-bit modes.

Some special PNG formats are broken.

All output images are converted to 8-bit RGB/RGBA.

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

With TTA, half float, minibatch
```
python -m waifu2x.cli --tta --amp --batch-size 16 -m scale -i tmp/images -o tmp/out
```
