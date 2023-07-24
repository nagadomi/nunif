## 0. Notes

Basically, I am working on Linux. I sometimes check to make sure that this code works in Windows.

I am not familiar with Windows or Anaconda. If you are familiar with Anaconda, do it your way.

## Easy way to install

### 1. Install Python

Install Python 3.10 from Windows Store.

### 2. Run installer

Run `Installer for Windows.bat`.

If you want to reinstall, delete `venv` folder and run it again.

If you have Anaconda installed, make sure you have added `python` to PATH.
If `python` command opens Windows Store, see https://stackoverflow.com/questions/58754860/cmd-opens-windows-store-when-i-type-python.

### 3.2 Run waifu2x GUI

Run `Run waifu2x GUI.bat`.

### 3.2 Run waifu2x web interface

Run `Run waifu2x Web.bat`.

Open http://localhost:8812/

### 3.2 Open CLI Prompt

Run `Open Prompt.bat`.

## Manually install

### 1. Install dependencies packages (Optional)

#### Install ImageMagick

NOTE: ImageMagick(wand) is only required for training and benchmark.

See [Install ImageMagick on Windows](https://docs.wand-py.org/en/0.6.10/guide/install.html?highlight=windows#install-imagemagick-on-windows).

#### Install libraqm.dll

NOTE: libraqm is only required for synthetic training data generation.

Download `libraqm‑0.7.1.dll.zip` from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow .

See https://stackoverflow.com/questions/62939101/how-to-install-pre-built-pillow-wheel-with-libraqm-dlls-on-windows

### 2. Clone

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

### 3. Setup venv (optional)

```
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Pytorch

```
pip install -r requirements-torch.txt
```

### 5. Install pip packages

```
pip install -r requirements.txt
```

### 6. Run waifu2x.web

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

If you got `ImportError: cannot import name '_imagingcms' from 'PIL'` error, upgrade the pillow package.
```
pip install --upgrade pillow
```

This seems to be a problem with the pillow that is installed by conda by default.

See also [waifu2x README.md](waifu2x/README.md).
