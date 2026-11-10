import os
import subprocess
import sys

import pytest

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TORCH_ENV_KEYS = (
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
)


def _run_python(code: str, env_overrides: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    for key in TORCH_ENV_KEYS:
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("module_name", ["nunif", "iw3", "stlizer", "cliqa", "dino", "create_training_data"])
def test_init_torch_env_before_torch_import(module_name: str) -> None:
    code = f"""
import os, sys
import {module_name}
assert "torch" not in sys.modules
assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
assert os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1"
"""
    _run_python(code)


@pytest.mark.parametrize(
    "module_name",
    [
        "waifu2x",
        "imagenet",
        "iw3.cli",
        "waifu2x.cli",
        "stlizer.cli",
        "nunif.cli.segment_video",
        "cliqa.filter_color",
        "imagenet.models.torchvision_models",
        "dino.models.l4sn",
        "train",
    ],
)
def test_init_torch_env_set_before_torch_loaded(module_name: str) -> None:
    code = f"""
import os, sys
orig_setdefault = os.environ.setdefault
events = []
def tracked_setdefault(key, value):
    if key in {TORCH_ENV_KEYS!r}:
        events.append((key, "torch" in sys.modules))
    return orig_setdefault(key, value)
os.environ.setdefault = tracked_setdefault
import {module_name}
assert events, "init_torch_env was not called"
assert all(not loaded for _, loaded in events), events
assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
assert os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1"
"""
    _run_python(code)


def test_init_torch_env_respects_existing_env() -> None:
    code = """
import os
import nunif
assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "0"
"""
    _run_python(
        code,
        env_overrides={
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0",
        },
    )


def test_nunif_lazy_submodules() -> None:
    code = """
import sys
import nunif

assert "torch" not in sys.modules
expected = {"models", "modules", "transforms", "training", "initializer"}
assert expected.issubset(set(nunif.__all__))
assert expected.issubset(set(dir(nunif)))

try:
    _ = nunif.non_existent_attr
except AttributeError:
    pass
else:
    raise AssertionError("AttributeError was not raised")

for name in expected:
    mod = getattr(nunif, name)
    assert mod.__name__ == f"nunif.{name}"
"""
    _run_python(code)
