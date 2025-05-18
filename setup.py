import setuptools
import os

# Read the long description from README.md (if available)
long_description = ""
if os.path.isfile("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

# Read requirements from requirements.txt
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as fh:
    for line in fh:
        # Trim whitespace and skip comments/empty lines
        line = line.strip()
        if line and not line.startswith("#"):
            # Remove inline comments and append
            requirements.append(line.split("#")[0].strip())

setuptools.setup(
    name="nunif",
    version="0.1.0",
    author="nagadomi",
    author_email="",
    description="Misc; latest version of waifu2x; 2D video to stereo 3D video conversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nagadomi/nunif",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)