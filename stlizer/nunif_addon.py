from nunif.addon import Addon
from . import models # noqa


def addon_config():
    return [OutpaintAddon()]


class OutpaintAddon(Addon):
    def __init__(self):
        super().__init__("stlizer.outpaint")

    def register_train(self, subparsers, default_parser):
        from .training.outpaint.trainer import register
        return register(subparsers, default_parser)
