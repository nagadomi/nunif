# Installation for Intel GPUs

First of all, I have confirmed that this software works on Intel Max 1100 on Intel Tiber AI Cloud.
However, I don't own any Intel GPU hardware myself, so I haven’t set it up myself.
(The setup has already been completed on Intel Tiber AI Cloud.)

There are two main differences compared to a standard installation.

1. Install Intel GPUs Driver
2. Install PyTorch for Intel GPUs (xpu)


## 1. Install Intel GPU Driver

See https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu.html

There is no need to install "Intel® Deep Learning Essentials".

## 2. Install PyTorch for Intel GPUs (xpu)

Use `requirements-torch-xpu.txt` instead of `requirements-torch.txt`.

```
pip3 install -r requirements-torch-xpu.txt
```

If you are using nunif-windows-package, change the following line in `update.bat`.

```
python -m pip install --no-cache-dir --upgrade -r "%NUNIF_DIR%\requirements-torch.txt"
```
to
```
python -m pip install --no-cache-dir --upgrade -r "%NUNIF_DIR%\requirements-torch-xpu.txt"
```

After that, run `update.bat`.
