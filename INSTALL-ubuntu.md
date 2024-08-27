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

## 4. Install Pytorch and pip packages

```
pip3 install -r requirements-torch.txt
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

If want to use NVENC(`h264_nvenc` and `hevc_nvenc`) in iw3, install PyAV from source code.
```
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
pip3 install av --force-reinstall --no-binary av
```

## 5. Download pre-trained models

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
