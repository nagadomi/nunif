from nunif.addon import Addon


def addon_config():
    return ImageNetAddon()


class ImageNetAddon(Addon):
    def __init__(self):
        super().__init__("imagenet")

    def register_train(self, subparsers, default_parser):
        from .training.trainer import register
        return register(subparsers, default_parser)
