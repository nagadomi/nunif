# Using `torch.compile` on Windows

This guide explains how to enable and use `torch.compile` in a Windows environment.  
Using `torch.compile` may improve the execution speed of some models, but the first run will take additional time due to compilation.

## Common Steps for GPU and CPU

All installation scripts are located in the `torch_compile` folder. If the folder does not exist, run `update-installer.bat`.

1. Run `enable_long_path.reg`
2. Restart your computer
3. Run `install_python_dev.bat`

### `enable_long_path.reg` – Enable Long File Paths

By default, Windows limits the maximum file path length to 260 characters.  
Running `enable_long_path.reg` and restarting your computer removes this limitation.

For more details, see Microsoft’s documentation:  
[Maximum file path limitation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry)

### `install_python_dev.bat` – Install Python Development Files

The **Embeddable Python** used by `nunif-windows` does not include development headers or libraries.  
This script installs the required development files into the environment.

## GPU

`torch.compile` for GPU on Windows is **not officially supported** by PyTorch or Triton.  
Use the Windows port of Triton available at the following repository:  
<https://github.com/woct0rdho/triton-windows>

The installation script is located in the `torch_compile` folder.

1. Run `install_triton_windows.bat`

After installation, select **NVIDIA GPU** as the device and verify that the `torch.compile` checkbox can be enabled.

Compilation caches are stored in the following directory:

`C:\Users<username>\AppData\Local\Temp\torchinductor_*`

You can safely delete this folder if you no longer need the cache.

## CPU

To use `torch.compile` on CPU, you must install **Visual Studio 2022** or **2019**.
Below is the recommended setup for Visual Studio 2022 Community Edition.

Download link:  
https://aka.ms/vs/17/release/vs_community.exe

1. Select **Desktop development with C++**
2. In the **Language packs** tab, select **English** (required)
3. Proceed with the installation

After installation, select **CPU** as the device and verify that the `torch.compile` checkbox can be enabled.

If it stops working after updates, try deleting the compilation cache and run it again:

`C:\Users<username>\AppData\Local\Temp\torchinductor_*`
