from nunif.addon import Addon
from . import models # noqa


def addon_config():
    return SBS3DAddon()


class SBS3DAddon(Addon):
    def __init__(self):
        super().__init__("sbs")

    def register_create_training_data(self, subparsers, default_parser):
        from .training.sbs.create_training_data import register
        return register(subparsers, default_parser)

    def register_train(self, subparsers, default_parser):
        from .training.sbs.trainer import register
        return register(subparsers, default_parser)
