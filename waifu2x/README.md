# Waifu2x

Download pre-trained models.
```
python -m waifu2x.download_models
```
This command downloads the pre-trained models to `waifu2x/pretrained_models`.

# Current limitations

The CLI and Web App allow you to switch the image library to be used with `--image-lib (pil|wand)`. When `--image-lib pil`(default), 16bit image output is not supported. If you want to use `--depth 16`, specify `--image-lib wand`.

When `--format jpeg` is specified, the transparent areas of the image will be set to white background and the alpha channel will be removed.

When `--format jpeg` or `--format webp` is specified, image-gamma value(gAMA) is ignored (removed from the output image).

# Note

In Ubuntu 22.04, gThumb(Image Viewer) has a bug in rendering webp's alpha channel. If you encounter any problems with the alpha channel, please check it with Google-chrome.

# Command Line Interface

See [waifu2x Command Line Interface](docs/cli.md).

# Web Application (server)

See [waifu2x Web Application](docs/web.md)

# Web Application (in-browser)

See [unlimited:waifu2x](unlimited_waifu2x/)

# Training

See [waifu2x Model Training](docs/training.md)
