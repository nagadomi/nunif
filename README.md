My playground.

For the time being, I will make incompatible changes.

## waifu2x

[waifu2x/README.md](./waifu2x/README.md)

waifu2x: Image Super-Resolution for Anime-Style Art. Also it supports photo models (GAN based models)

The repository contains waifu2x pytorch implementation and pretrained models, started with porting the original [waifu2x](https://github.com/nagadomi/waifu2x).

The demo application can be found at
- https://waifu2x.udp.jp/ (Cloud version)
- https://unlimited.waifu2x.net/ (In-Browser version).

## iw3

[iw3/README.md](./iw3/README.md)

I want to watch any 2D video as 3D video on my VR device, so I developed this very personal tool.

iw3 provides the ability to convert any 2D image/video into side-by-side 3D image/video. However, it does not support Anime.

## cliqa

[cliqa/README.md](./cliqa/README.md)

`cliqa` provides low-vision image quality scores that are more consistent across different images.

It is useful for filtering low-quality images with a threshold value when creating image datasets.

Currently, the following two models are supported.

- JPEGQuality: Predicts JPEG Quality from image content
- GrainNoiseLeve: Predicts Noise Level related to photograph and PSNR degraded by that noise

CLI tools are also available to filter out low quality images using these results.

## Installer for non-developer Windows users

- [nunif windows package](windows_package/docs/README.md)
- [nunif windows package (日本語)](windows_package/docs/README_ja.md)

## For developers

### Dependencies

- Python 3 (Probably works with Python 3.9 or later, developed with 3.10)
- [PyTorch](https://pytorch.org/get-started/locally/)
- See requirements.txt

We usually support the latest version. If there are bugs or compatibility issues, we will specify the version.

- [INSTALL-ubuntu](INSTALL-ubuntu.md)
- [INSTALL-windows](INSTALL-windows.md)
