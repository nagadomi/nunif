import argparse
import os
import zipfile
from datetime import datetime

from ..utils.git import get_last_commit_date

EXCLUDE_DIRS = {"python", "git", "nunif", "__pycache__", ".git"}
DEFAULT_INPUT_DIR = "windows_package"
DEFAULT_OUTPUT_DIR = "tmp"


def get_default_output_file() -> str:
    commit_date = get_last_commit_date() or datetime.now()
    return os.path.join(DEFAULT_OUTPUT_DIR, f"nunif-windows-{commit_date.strftime('%Y%m%d')}.zip")


def iter_package_paths(root: str) -> list[str]:
    """Collect sorted paths under root, excluding runtime and VCS directories."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Input directory not found: {root}")

    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        if dirpath != root:
            paths.append(dirpath)
        for name in sorted(filenames):
            if name.endswith(".zip"):
                continue
            paths.append(os.path.join(dirpath, name))
    return paths


def to_arcname(path: str, root: str) -> str:
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    if os.path.isdir(path):
        return f"{rel}/"
    return rel


def pack_windows_package(
    input_dir: str,
    output_file: str,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Package windows_package contents into a zip archive without the parent directory."""
    paths = iter_package_paths(input_dir)
    arcnames = [to_arcname(path, input_dir) for path in paths]

    if dry_run:
        return arcnames

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in zip(paths, arcnames, strict=True):
            if os.path.isdir(path):
                zf.write(path, arcname=arcname)
            else:
                zf.write(path, arcname=arcname, compress_type=zipfile.ZIP_DEFLATED)
    return arcnames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package windows_package into nunif-windows.zip without the parent directory."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=f"Output zip file path (default: {os.path.join(DEFAULT_OUTPUT_DIR, 'nunif-windows-YYYYMMDD.zip')})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List archive entries without creating the zip file.",
    )
    args = parser.parse_args()

    output_file = args.output if args.output is not None else get_default_output_file()
    arcnames = pack_windows_package(args.input, output_file, dry_run=args.dry_run)
    for name in arcnames:
        print(name)
    if args.dry_run:
        print(f"Dry run: {len(arcnames)} entries would be written to {output_file}")
    else:
        print(f"Created {output_file} ({len(arcnames)} entries)")


if __name__ == "__main__":
    main()
