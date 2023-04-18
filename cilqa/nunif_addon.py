from nunif.addon import Addon
from . import models # noqa


def addon_config():
    return [CILQAJPEGNoiseAddon(), CILQAGrainNoiseAddon()]


class CILQAJPEGNoiseAddon(Addon):
    def __init__(self):
        super().__init__("cilqa.jpeg")

    def register_train(self, subparsers, default_parser):
        from .training.jpeg_noise_trainer import register
        return register(subparsers, default_parser)


class CILQAGrainNoiseAddon(Addon):
    def __init__(self):
        super().__init__("cilqa.grain")

    def register_train(self, subparsers, default_parser):
        from .training.grain_noise_trainer import register
        return register(subparsers, default_parser)
