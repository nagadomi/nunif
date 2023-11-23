# Python API (torch.hub)

## Dependency modules

`torch`, `torchvision`, `packaging`

## Overview

```python
import torch
from PIL import Image

model = torch.hub.load("nagadomi/nunif:master", "waifu2x",
                       method="scale", noise_level=3, trust_repo=True).to("cuda")
input_image = Image.open("input.jpg")
result = model.infer(input_image)
result.show() # result is PIL.Image.Image
```

1. loading model with `torch.hub.load()`
2. converting image with `model.infer(x)` or `model(x)`

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

### troch.hub.load from local source

Specify `source="local"` option. And specify the nunif directory(this repository) for the first argument.
```
model = torch.hub.load("./nunif", "waifu2x",
                       method="scale", noise_level=3, source="local", trust_repo=True).cuda()
```

Also you can directly import `waifu2x/hub.py:waifu2x()`.
```
import sys
sys.path.append("./nunif")
from waifu2x.hub import waifu2x

model = waifu2x(model_type="art").cuda()
```

### Using fp16 model manually without AMP

You can convert the model to fp16 by using `half()` method.

```
model = torch.hub.load("nagadomi/nunif:master", "waifu2x",
                       method="scale", noise_level=3, amp=False, trust_repo=True).cuda().half()
```

If you pass a Tensor as input, call `cuda().half()` manually.
```
z = model(x.cuda().half())
```
Note that the fp16 model does not currently work with CPU device.

## `model.convert(input_filepath, output_filepath, tta=False, format="png")`

This is the only method that preserves the ICC Profile.

## `model.infer(x, tta=False, output_type="pil")`

`x` can accept the following 3 data types.
- `PIL.Image.Image`
- `str`: Handled as a image file path
- `torch.Tensor`: Handled as a float32 CHW RGB image

When `output_type="tensor"` is specified, returns a tuple of tensor `(rgb, alpha)`.

When specifying x as a tensor, it is safe to convert it to float32 and transfer it to `model.device` in advance.
The following method works when x in (cuda float32 tensor, cuda float16 tensor, cpu float16 tensor, cuda float32 tensor).

```
ret = model.infer(x.float().to(model.device))
```

## Reloading source code

When specifying a remote repository with `torch.hub.load`, the source code and pretrained models are stored in `~/.cache/torch/hub`.
(You can change the directory with `torch.hub.set_dir()`.)

You can use the following method (`force_reload=True` option) to force update to the latest source code.
```
torch.hub.help("nagadomi/nunif:master", "waifu2x", force_reload=True, trust_repo=True)
```
Note that `force_reload=True` will also re-download the pretrained models. This option is not recommended except for maintenance purposes.


For more details, see [../hub.py](../hub.py).
