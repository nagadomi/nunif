import os
import subprocess
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run(cmd: list[str], cwd: str | None = None) -> None:
    args = list(cmd)

    if args and args[0] == "python":
        args[0] = sys.executable

    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=cwd or PROJECT_DIR)


def run_module(module_name: str, cwd: str | None = None) -> None:
    run(["python", "-m", module_name], cwd=cwd)
