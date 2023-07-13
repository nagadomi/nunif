import os
import sys
import torch


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
