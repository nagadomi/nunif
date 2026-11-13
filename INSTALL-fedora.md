# Installing nunif on Fedora Linux

This guide provides step-by-step instructions for installing `nunif` and its dependencies on Fedora Linux.

## Prerequisites

* You will need `sudo` privileges to install system-wide packages.
* A stable internet connection is required for downloading packages and model files.

## Important: RPM Fusion Repositories for Multimedia Support

For full multimedia functionality, particularly for `ffmpeg` (used by `PyAV` for video processing), you will likely need to enable the RPM Fusion repositories. These repositories provide software that Fedora cannot ship due to licensing or patent reasons.

If you haven't enabled RPM Fusion already, run the following commands in your terminal:

```bash
sudo dnf install -y \
  [https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm](https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm) -E %fedora).noarch.rpm
sudo dnf install -y \
  [https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm](https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm) -E %fedora).noarch.rpm

# Optional: Update your package metadata after adding new repositories
sudo dnf check-update
```

## Installation Steps

### 1. Install System Dependencies

Open your terminal and install the necessary system libraries:

```bash
sudo dnf install -y git ImageMagick-devel libraqm-devel
```
* `git`: For cloning the `nunif` repository.
* `ImageMagick-devel`: Development files for ImageMagick, used for image processing tasks.
* `libraqm-devel`: Development files for Raqm, a library for complex text layout.

### 2. Clone the nunif Repository

Clone the `nunif` source code from GitHub:

```bash
git clone [https://github.com/nagadomi/nunif.git](https://github.com/nagadomi/nunif.git)
cd nunif
```

If you want to use the `dev` branch (for the latest, potentially unstable, features):
```bash
# Option 1: Clone directly
git clone [https://github.com/nagadomi/nunif.git](https://github.com/nagadomi/nunif.git) -b dev
cd nunif # Navigate into the cloned directory if you used this option
```
or, if you've already cloned the main branch and are inside the `nunif` directory:
```bash
# Option 2: Switch to dev branch
git fetch --all
git checkout -b dev origin/dev
```

### 3. Set Up a Python Virtual Environment (Highly Recommended)

Using a Python virtual environment is highly recommended to isolate `nunif`'s dependencies from your system's global Python packages. This prevents potential conflicts.

Initialize the virtual environment (e.g., named `.venv` within the `nunif` directory):
```bash
python3 -m venv .venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```
Your terminal prompt should change to indicate that the virtual environment is active (e.g., `(.venv) user@host:...$`).

*(To deactivate the virtual environment later, simply run `deactivate` from within the environment.)*

### 4. Install Python Dependencies (PyTorch and others)

Ensure your virtual environment is activated. Then, install the required Python packages using `pip`:

```bash
pip3 install -r requirements-torch.txt
pip3 install -r requirements.txt
```

### 5. Install GUI Dependencies (Optional)

These packages are only required if you intend to use the GUI components of `nunif`.

**First, ensure GTK3 development files are installed if you plan to build wxPython from source or if it's a dependency for a pre-built wheel:**
```bash
sudo dnf install -y gtk3-devel
```

**Then, install wxPython (choose one method):**

* **Method 1: Install wxPython via pip (Recommended for flexibility)**
    This method often provides more up-to-date versions and works well within virtual environments.
    ```bash
    pip3 install -f [https://extras.wxpython.org/wxPython4/extras/linux/gtk3/](https://extras.wxpython.org/wxPython4/extras/linux/gtk3/) wxpython
    ```
    *(The URL points to general Linux GTK3 builds; it should generally work on Fedora.)*

* **Method 2: Try installing wxPython via dnf (System package)**
    The package name might vary depending on your Fedora version. You can search for it using `sudo dnf search wxpython`.
    ```bash
    # Example package name, this might differ or not be available
    sudo dnf install -y python3-wxpython4
    ```

* **Method 3: Build wxPython from source using `requirements-gui.txt`**
    This uses the `requirements-gui.txt` file, which typically lists wxPython as a dependency to be built.
    ```bash
    pip3 install -r requirements-gui.txt
    ```

### 6. Building PyAV from Source (Conditional)

The binary wheels for `PyAV` (installed via `pip3 install av`) often include common codecs. However, you might need to build `PyAV` from source if:
* You need to use a specific version of FFmpeg available on your system (e.g., the one from RPM Fusion with more codecs enabled).
* You require codecs not included in the pre-built `PyAV` wheels.
* You specifically need an LGPL version of FFmpeg.

The `nunif` README notes that NVENC support is included in PyAV binary packages from version 14.2.0, potentially reducing the need to build from source just for NVENC.

**If you need to build PyAV from source:**

1.  **Install FFmpeg Development Libraries:**
    Ensure RPM Fusion is enabled (see "Important" note at the beginning of this guide).
    ```bash
    sudo dnf install -y ffmpeg-devel
    ```
    The `ffmpeg-devel` package from RPM Fusion should include all necessary development headers and libraries (like `libavcodec-devel`, `libavformat-devel`, etc.).

2.  **Install PyAV from source using pip:**
    Choose the PyAV version compatible with your FFmpeg version (refer to `nunif` or PyAV documentation for specifics if unsure):
    * `av==13.1.0` generally works with FFmpeg 6.x.
    * `av==14.3.0` generally works with FFmpeg 7.1.x.

    ```bash
    # Example for FFmpeg 6.x compatibility
    pip3 install av==13.1.0 --force-reinstall --no-binary av

    # Example for FFmpeg 7.x compatibility (uncomment if needed)
    # pip3 install av==14.3.0 --force-reinstall --no-binary av
    ```

    You can also install directly from a specific Git commit/tag if needed:
    ```bash
    # Example for a specific tag (uncomment and adjust as needed)
    # pip3 install --force-reinstall git+[https://github.com/PyAV-Org/PyAV.git@v13.1.0](https://github.com/PyAV-Org/PyAV.git@v13.1.0)
    # pip3 install --force-reinstall git+[https://github.com/PyAV-Org/PyAV.git@v14.3.0](https://github.com/PyAV-Org/PyAV.git@v14.3.0)
    ```

3.  **Using a Custom FFmpeg Path (Advanced):**
    If you have compiled FFmpeg in a custom location (e.g., `~/opt/ffmpeg/`), you can tell PyAV where to find its `pkg-config` files:
    ```bash
    PKG_CONFIG_LIBDIR=~/opt/ffmpeg/lib/pkgconfig pip3 install --force-reinstall git+[https://github.com/PyAV-Org/PyAV.git@v14.3.0](https://github.com/PyAV-Org/PyAV.git@v14.3.0)
    ```
    At runtime, you'll also need to set `LD_LIBRARY_PATH` so the system can find the shared libraries:
    ```bash
    export LD_LIBRARY_PATH=~/opt/ffmpeg/lib:$LD_LIBRARY_PATH
    # If needed, also add its binaries to your PATH for command-line access
    # export PATH=~/opt/ffmpeg/bin:$PATH
    ```
    Binary builds of FFmpeg with various configurations can be found at [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) (ensure you download a `*-shared` version if you intend to link against it when building PyAV).

### 7. Download Pre-trained Models

`nunif` requires pre-trained models for its various functionalities. These commands should be run from the root of the `nunif` directory.

* **For waifu2x:**
    ```bash
    python3 -m waifu2x.download_models
    python3 -m waifu2x.web.webgen # For the web interface
    ```
    For more details, see `waifu2x/README.md`.

* **For iw3:**
    ```bash
    python3 -m iw3.download_models
    ```
    For more details, see `iw3/README.md`.

## Running nunif

After installation, you can typically run `nunif` tools using `python3 -m nunif.<module_name>` or specific scripts provided within the repository. Refer to the main `README.md` of the `nunif` project for detailed usage instructions for its different components.

Example (this is a generic example, the actual command will depend on the specific `nunif` tool you want to use):
```bash
# Ensure your virtual environment is active
# source .venv/bin/activate # If not already active

# Example placeholder for running a CLI tool, replace with actual command
# python3 -m nunif.cli --input <your_image.png> --output <processed_image.png> ...
```

## Troubleshooting

* **ImportError or Codec Issues:** If you encounter issues related to missing codecs or multimedia libraries (especially with video processing), ensure RPM Fusion is enabled and that `ffmpeg` and `ffmpeg-devel` are correctly installed from there. If you built PyAV from source, double-check that it linked against the intended FFmpeg version.
* **`Command not found` (for `pip3`, `python3`):** Ensure Python 3 and its pip module are correctly installed on your Fedora system. The command `sudo dnf install python3 python3-pip` should provide them.
* **Virtual Environment:** Always make sure your virtual environment is activated (`source .venv/bin/activate`) before running `pip install` commands or executing `nunif` scripts. This ensures you are using the correct Python interpreter and its associated packages.
* **Permissions:** If you encounter permission errors when installing system packages, ensure you are using `sudo`. For `pip` installations within an activated virtual environment, `sudo` is generally not needed and should be avoided.

---
This guide should help Fedora users get `nunif` up and running. For issues specific to `nunif` functionality, please refer to the official `nunif` documentation or issue tracker.
