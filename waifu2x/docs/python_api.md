# Python API (torch.hub)

## Dependency modules

`torch`, `torchvision`, `packaging`

## Overview

```python
model = torch.hub.load("nagadomi/nunif:master", "waifu2x",
                       method="scale", noise_level=3, trust_repo=True).to("cuda")
input_image = PIL.Image.open("input.jpg")
result = model.infer(input_image)
result.show() # result is PIL.Image.Image
```

1. loading model with `torch.hub.load()`
2. converting image with `model.infer()`

## `torch.hub.load("nagadomi/nunif:master", "waifu2x", ...)`

```python
def waifu2x(model_type="art",
            method=None, noise_level=-1,
            device_ids=[-1], tile_size=256, batch_size=4, keep_alpha=True, amp=True,
            **kwargs):
```

| Arguments    | Description
|--------------|---------------
| `model_type` | `art`, `art_scan`, `photo`, see `MODEL_TYPES` in [../hub.py](../hub.py)
| `method`     | `noise`: 1x denoising, `scale` or `scale2x`: 2x, `scale4x`: 4x.
| `noise_level`| -1: none, 0-3: denoising level.
| `device_ids` | When specified, the model will be loaded into the specified CUDA device. If you want to use multiple GPUs, specify multiple GPU id. When not specified, the model will be loaded into CPU. After loading, it should be loaded into CUDA using `model.to(device)`.
| `tile_size`    | tile size
| `batch_size`   | batch size
| `keep_alpha`   | When `False` is specified, the alpha channel will be dropped.
| `amp`          | When `False` is specified, performs in FP32 mode. FP16 by defualt.

When `method` is not specified(None), models of all `method` and `noise_level` will be loaded.
In this case, you must specify `method` and `noise_level` using `model.set_mode(method, noise_level)` before executing `model.infer()` or specify `method` and `noise_level` argument for `infer()`.
example,
```python
import torch
from PIL import Image
import threading

lock = threading.Lock()
im = Image.open("input.jpg")
model = torch.hub.load("nagadomi/nunif:master", "waifu2x",
                       model_type="art_scan", trust_repo=True).to("cuda")
for noise_level in (-1, 0, 1, 2, 3):
    with lock: # Note model.set_mode -> model.infer block is not thread-safe
        # Select method and noise_level
        model.set_mode("scale", noise_level)
        out = model.infer(im)
    out.save(f"noise_scale_{noise_level}.png")
    # or specify method and noise_level arguments
    out = model.infer(im, method="noise", noise_level=noise_level)
    out.save(f"noise_{noise_level}.png")
```

## `model.convert(input_filepath, output_filepath, tta=False, format="png")`

This is the only method that preserves the ICC Profile.

## `model.infer(x, tta=False, output_type="pil")`

`x` can accept the following 3 data types.
- `PIL.Image.Image`
- `str`: Handled as a image file path
- `torch.Tensor`: Handles as float32 CHW RGB image

When `output_type="tensor"` is specified, returns a tuple of tensor `(rgb, alpha)`.


For more details, see [../hub.py](../hub.py).
