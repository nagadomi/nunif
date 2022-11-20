## 0. Notes

Basically, I am working on Linux. I will check the work with Windows sometimes.

I am not familiar with Windows or Anaconda. If you are familiar with Anaconda, do it your way.

## 1. Install dependencies packages

TODO: ImageMagick (Not required to run waifu2x.web)

Perhaps, on windows you need to install the ImageMagick Binary Package, but I haven't tested it.

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
conda install python=3.9
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

## 6. Run waifu2x.web

Generate `waifu2x/public_html`
```
python -m waifu2x.webgen.gen
```

Start the web server.
```
python -m waifu2x.web
```
Open http://localhost:8812/

If you don't have an nvidia GPU, specify the `--gpu -1` option. (CPU Mode)
```
python -m waifu2x.web --gpu -1
```

If you got `ImportError: cannot import name '_imagingcms' from 'PIL'` error, upgrade the pillow package.
```
pip3 install --upgrade pillow
```

This seems to be a problem with the pillow that is installed by conda by default.
