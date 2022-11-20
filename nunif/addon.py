import os
from os import path

# TODO: train


class Addon():
    def __init__(self, name, **kwargs):
        self.config = {"name": name}
        self.config.update(kwargs)

    def name(self):
        return self.config["name"]


def load_addon(addon_dir):
    addon_py = path.join(addon_dir, "nunif_addon.py")
    addon_module_path = path.splitext(path.relpath(addon_py))[0].replace(os.sep, ".")
    addon_module = __import__(addon_module_path, globals(), fromlist=["addon_config"])

    return addon_module.addon_config()


def load_addons(addon_dirs):
    addons = []
    for addon_dir in addon_dirs:
        addons.append(load_addon(addon_dir))
    return addons
