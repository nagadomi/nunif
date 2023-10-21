try:
    import packaging as _packaging
except ImportError:
    raise RuntimeError("Missing dependencies `packaging`. Try `pip install packaging`")
from waifu2x.hub import waifu2x

dependencies = ["torch", "torchvision", "packaging"]
