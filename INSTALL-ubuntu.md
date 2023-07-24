## 1. Install dependencies packages

```
sudo apt-get install git-core libmagickwand-dev libraqm-dev
```

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

```
pip3 install -r requirements-torch.txt
```

## 5. Install pip packages

```
pip3 install -r requirements.txt
```

If you want to use GUI, install wxpython >= 4.0.0
```
sudo apt-get install python3-wxgtk4.0
```
or install wheel package from https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ .
```
pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04/ wxpython
```
or build from source. See [Prerequisites](https://github.com/wxWidgets/Phoenix#prerequisites) first.
```
sudo apt-get install libgtk-3-dev
pip3 install -r requirements-gui.txt
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

See also [waifu2x README.md](waifu2x/README.md).
