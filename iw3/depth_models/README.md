# iw3/depth_models

This directory contains the depth estimation model definitions available in iw3.

- Python files (`.py`) placed in this directory are automatically loaded.
- Filenames starting with `_` are ignored by `.gitignore`. Using a leading `_` is recommended for user-defined models or extensions.
- To add a depth estimation model to iw3, define a subclass of `BaseDepthModel` (defined in `../base_depth_model.py`). Refer to `../base_depth_model.py` and existing implementations in this directory for details.
