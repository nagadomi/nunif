from nunif.addon import Addon
from . import models # noqa


def addon_config():
    return [CILQAAddon()]


class CILQAAddon(Addon):
    def __init__(self):
        super().__init__("cilqa.jpeg")

    def register_train(self, subparsers, default_parser):
        from .training.jpeg_trainer import register as jpeg_register
        return jpeg_register(subparsers, default_parser)
