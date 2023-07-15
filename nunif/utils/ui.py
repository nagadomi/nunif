import os
from os import path
import sys
import torch
import mimetypes


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class TorchHubDir:
    def __init__(self, hub_dir):
        self.hub_dir = hub_dir
        self.original_hub_dir = None

    def __enter__(self):
        self.original_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(self.hub_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.hub.set_dir(self.original_hub_dir)


def is_image(filename):
    mime = mimetypes.guess_type(filename)[0]
    return mime and mime.startswith("image")


def is_video(filename):
    mime = mimetypes.guess_type(filename)[0]
    return mime and mime.startswith("video")


def is_text(filename):
    mime = mimetypes.guess_type(filename)[0]
    return mime and mime.startswith("text")


def is_output_dir(filename):
    return path.isdir(filename) or "." not in path.basename(filename)


def make_parent_dir(filename):
    parent_dir = path.dirname(filename)
    if not path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
