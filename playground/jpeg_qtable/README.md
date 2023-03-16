# JPEG Quantization Table

Typically, JPEG quality is specified as a number with a range of 0-100.
With Pillow, QTable can be specified directly instead of the quality value.

That means that JPEG artifacts can be controlled, which is useful for creating the dataset to be used for denoising poor quality JPEG encoders.

This repository contains the following code.

- `show.py`: Show QTable of the specified JPEG file.
- `make_extreme_jpeg.py`: A sample code to create a JPEG file with customized QTable.
- `search_qtable.py`: Search for QTables that produce checkerboard and stripe patterns using Hill-Climbing method.
- `collect_qtable.py`: Save `*_qtable.json` created by `search_qtable.py` as a single file in pth format.
