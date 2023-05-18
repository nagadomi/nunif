## 0. Limitations

Currently, it works very limitedly on macOS. (2023-05-18)

`swin_unet/*` models (waifu2x's default models) only work with PyTorch 2.0.1 nightly build.
However, even with PyTorch 2.0.1, `swin_unet/art` output glitches, and `swin_net/art_scan` and `swin_net/photo` do not work on MPS.

Practically, only the older models `upcunet`, `cunet`, `upconv_7`, and `vgg_7` will work.

To use CUNet models,
- Specify `--model-dir waifu2x/pretrained_models/cunet/art` option for `waifu2x.cli`
- Specify `--art-model-dir waifu2x/pretrained_models/cunet/art` option for `waifu2x.web`

Also PyTorch's MPS device does not support 16-bit float operation. If you get errors related to `autocast`, specify `--disable-amp` option or use `--gpu -1`(CPU only mode).

## 1. Install dependencies packages

Install Python 3.10+ (3.10+ is required for macOS. related to https://github.com/urllib3/urllib3/issues/2168 )

```
brew install python3
```

Install ImageMagick and libraqm.
```
brew install imagemagick libraqm
```

## 2. Clone

```
git clone https://github.com/nagadomi/nunif.git
cd nunif
```

## 3. Setup virtualenv (optional)

initialize
```
python3 -m venv .venv
```

activate
```
source .venv/bin/activate
```

(exit)
```
deactivate
```

## 4. Install Pytorch

See [Pytorch](https://pytorch.org/get-started/locally/)

```
pip3 install torch torchvision torchaudio torchtext
```

## 5. Install pip packages

```
pip3 install -r requirements.txt
```

## 6. Run waifu2x.web

Download pre-trained models.
```
python -m waifu2x.download_models
```

Generate `waifu2x/web/public_html`
```
python -m waifu2x.web.webgen
```

Start the web server.
```
python -m waifu2x.web --art-model-dir ./waifu2x/pretrained_models/cunet/art --photo-model-dir ./waifu2x/pretrained_models/upconv_7/photo
```
Open http://localhost:8812/ (`style=art_scan` does not work)

See also [waifu2x README.md](waifu2x/README.md).
