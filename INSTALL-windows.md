## 0. Notes

Basically, I am working on Linux. I sometimes check to make sure that this code works in Windows.

I am not familiar with Windows or Anaconda. If you are familiar with Anaconda, do it your way.

## 1. Install dependencies packages (Optional)

#### Install ImageMagick

NOTE: ImageMagick(wand) is only required for training and benchmark.

See [Install ImageMagick on Windows](https://docs.wand-py.org/en/0.6.10/guide/install.html?highlight=windows#install-imagemagick-on-windows).

#### Install libraqm.dll

NOTE: libraqm is only required for synthetic training data generation.

Download `libraqm‑0.7.1.dll.zip` from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow .

See https://stackoverflow.com/questions/62939101/how-to-install-pre-built-pillow-wheel-with-libraqm-dlls-on-windows

## 2. Clone

```
git clone https://github.com/nagadomi/nunif.git
cd nunif
```

If you want to use the `dev` branch, execute the following command.
```
git clone https://github.com/nagadomi/nunif.git -b dev
```
or
```
git fetch --all
git checkout -b dev origin/dev
```

## 3. Setup conda env (optional)

```
conda create -n nunif
conda activate nunif
conda install python=3.10
```

## 4. Install Pytorch

See [Pytorch](https://pytorch.org/get-started/locally/)

```
conda install pytorch torchvision torchaudio torchtext pytorch-cuda=11.6 -c pytorch -c nvidia
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
python -m waifu2x.web
```
Open http://localhost:8812/

If you don't have an NVIDIA GPU, specify the `--gpu -1` option. (CPU Mode)
```
python -m waifu2x.web --gpu -1
```

If you got `ImportError: cannot import name '_imagingcms' from 'PIL'` error, upgrade the pillow package.
```
pip3 install --upgrade pillow
```

This seems to be a problem with the pillow that is installed by conda by default.



See also [waifu2x README.md](waifu2x/README.md).
