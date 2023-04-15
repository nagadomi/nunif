import os
from os import path
# TODO: train


class Addon():
    def __init__(self, name):
        self.name = name

    def name(self):
        return self.name

    def register_create_training_data(self, subparser, default_parser):
        pass

    def register_train(self, subparser, default_parser):
        pass


def load_addon(addon_dir):
    addon_py = path.join(addon_dir, "nunif_addon.py")
    if path.exists(addon_py):
        addon_module_path = path.splitext(path.relpath(addon_py))[0].replace(os.sep, ".")
        addon_module = __import__(addon_module_path, globals(), fromlist=["addon_config"])
        return addon_module.addon_config()
    else:
        return None


def load_addons(addon_dirs=None):
    if addon_dirs is None:
        search_dirs = [
            path.join(path.dirname(__file__), ".."),
            path.join(path.dirname(__file__), "..", "playground")]
        addon_dirs = []
        for root_dir in search_dirs:
            for subdir in os.listdir(root_dir):
                subdir = path.join(root_dir, subdir)
                if path.isdir(subdir):
                    addon_file = path.join(root_dir, subdir, "nunif_addon.py")
                    if path.exists(addon_file):
                        addon_dirs.append(subdir)

    addons = []
    for addon_dir in addon_dirs:
        addon = load_addon(addon_dir)
        if addon is not None:
            if isinstance(addon, (tuple, list)):
                for subaddon in addon:
                    addons.append(subaddon)
            else:
                addons.append(addon)
    return addons
