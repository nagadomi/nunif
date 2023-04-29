from nunif.addon import Addon


def addon_config():
    return Waifu2xAddon()


class Waifu2xAddon(Addon):
    def __init__(self):
        super(Waifu2xAddon, self).__init__("waifu2x")

    def register_create_training_data(self, subparsers, default_parser):
        from .training.create_training_data import register
        return register(subparsers, default_parser)

    def register_train(self, subparsers, default_parser):
        from .training.trainer import register
        return register(subparsers, default_parser)
