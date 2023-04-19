from nunif.addon import Addon
from . import models # noqa


def addon_config():
    return [CLIQAJPEGNoiseAddon(), CLIQAGrainNoiseAddon()]


class CLIQAJPEGNoiseAddon(Addon):
    def __init__(self):
        super().__init__("cliqa.jpeg")

    def register_train(self, subparsers, default_parser):
        from .training.jpeg_noise_trainer import register
        return register(subparsers, default_parser)


class CLIQAGrainNoiseAddon(Addon):
    def __init__(self):
        super().__init__("cliqa.grain")

    def register_train(self, subparsers, default_parser):
        from .training.grain_noise_trainer import register
        return register(subparsers, default_parser)
