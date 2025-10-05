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

### Pip packages
```
pip3 install -r requirements.txt
```
### Pytorch
If you are using an NVIDIA GPU, 
```
pip3 install -r requirements-torch.txt
```
or if you are using an AMD GPU.
```
pip3 install -r requirements-torch-rocm.txt
```
### GUI (optional)
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

For instructions on how to build PyAV from source, please refer to [Building PyAV from source](#building-pyav-from-source).

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


### Building PyAV from source

Since NVENC is included in the binary package from PyAV 14.2.0, there is no need to build from source if you just want to use NVENC.

If you want to use the LGPL version of ffmpeg or codecs that are not included in PyAV, you can build PyAV from source to link it with your local ffmpeg.

PyAV supports different FFmpeg versions depending on the PyAV version.
- `av==13.1.0` works with FFmpeg 6.x.x (I am not sure of the exact minor version).
- `av==14.3.0` works with FFmpeg 7.1.x.

When use local ffmpeg package.
```
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
pip3 install av==13.1.0 --force-reinstall --no-binary av
```

You can also specify a GitHub branch instead of using pip sdist.
```
pip3 install --force-reinstall git+https://github.com/PyAV-Org/PyAV.git@v13.1.0
```
```
pip3 install --force-reinstall git+https://github.com/PyAV-Org/PyAV.git@v14.3.0
```

If you want to specify the path to ffmpeg, you can specify it with `PKG_CONFIG_LIBDIR`. (In the case of `~/opt/ffmpeg/`, for example)
```
PKG_CONFIG_LIBDIR=~/opt/ffmpeg/lib/pkgconfig pip install --force-reinstall git+https://github.com/PyAV-Org/PyAV.git@v14.3.0
```
Note that in this case, setting `LD_LIBRARY_DIR` is required at runtime.
```
export LD_LIBRARY_PATH=~/opt/ffmpeg/lib:$LD_LIBRARY_PATH
# if needed
# export PATH=~/opt/ffmpeg/bin:$PATH
```

Binary builds of ffmpeg with various configurations and versions are available from https://github.com/BtbN/FFmpeg-Builds/releases .
(`*-shared` version is required to build.)
