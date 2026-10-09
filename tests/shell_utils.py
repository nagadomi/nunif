import os
import shlex
import subprocess
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def run(cmd: str | list[str], cwd: str | None = None) -> None:
    if isinstance(cmd, str):
        args = shlex.split(cmd)
    else:
        args = list(cmd)

    if args and args[0] == "python":
        args[0] = sys.executable

    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=cwd or PROJECT_DIR)


def run_script(script_path: str, cwd: str | None = None) -> None:
    print(f"=== Running: {script_path} ===", flush=True)
    subprocess.run([sys.executable, script_path], check=True, cwd=cwd or PROJECT_DIR)


def run_module(module_name: str, cwd: str | None = None) -> None:
    print(f"=== Running: {module_name} ===", flush=True)
    run(f"python -m {module_name}", cwd=cwd)
