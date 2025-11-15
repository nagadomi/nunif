## 0. Notes

Basically, I am working on Linux. I sometimes check to make sure that this code works in Windows.

I am not familiar with Windows or Anaconda. If you are familiar with Anaconda, do it your way.

## nunif windows pacakge

For Windows, I strongly recommend using the nunif windows package.

- [nunif windows package](windows_package/docs/README.md)

## Manually install

### 1. Install Python

Install Python 3.12 from Windows Store.

### 2. Install dependencies packages (Optional)

#### Install ImageMagick

NOTE: ImageMagick(wand) is only required for training and benchmark.

See [Install ImageMagick on Windows](https://docs.wand-py.org/en/0.6.10/guide/install.html?highlight=windows#install-imagemagick-on-windows).

### 3. Clone

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

### 4. Setup venv (optional)

```
python -m venv venv
.\venv\Scripts\activate
```


## 5. Install Pytorch and pip packages

### Pytorch

```
pip install -r requirements-torch.txt
```

### Pip packages
```
pip3 install -r requirements.txt
```

If you want to use GUI,
```
pip install -r requirements-gui.txt
```

## 6. Download pre-trained models

### waifu2x

```
python3 -m waifu2x.download_models
python3 -m waifu2x.web.webgen
```

See also [waifu2x README.md](waifu2x/README.md)

### iw3

```
python -m iw3.download_models
```

See also [iw3 README.md](iw3/README.md).

