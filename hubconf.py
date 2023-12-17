try:
    import packaging as _packaging
except ImportError:
    raise RuntimeError("Missing dependencies `packaging`. Try `pip install packaging`")

dependencies = ["torch", "torchvision", "packaging"]


def waifu2x(model_type="art",
            method=None, noise_level=-1,
            device_ids=[-1], tile_size=256, batch_size=4, keep_alpha=True, amp=True,
            **kwargs):
    from waifu2x.hub import waifu2x as _waifu2x
    return _waifu2x(model_type=model_type, method=method, noise_level=noise_level, device_ids=device_ids,
                    tile_size=tile_size, batch_size=batch_size, keep_alpha=keep_alpha, amp=amp,
                    **kwargs)
