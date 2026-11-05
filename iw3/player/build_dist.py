import argparse
import os
import zipfile
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_zip(source_dir, zip_path_template, arc_root, exclude_extensions=None):
    """Creates a zip file from source_dir, placing contents under arc_root.
    Returns the latest modification time of the files included.
    zip_path_template should contain {date} placeholder.
    """
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory not found: {source_dir}")
        return 0.0

    latest = 0.0
    file_entries = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith("~") or file.endswith(".bak") or file.startswith("."):
                continue
            if exclude_extensions and any(file.endswith(ext) for ext in exclude_extensions):
                continue
            abs_path = os.path.join(root, file)
            mtime = os.path.getmtime(abs_path)
            if mtime > latest:
                latest = mtime
            rel_path = os.path.relpath(abs_path, source_dir)
            arc_path = os.path.join(arc_root, rel_path)
            file_entries.append((abs_path, arc_path))

    if not file_entries:
        return 0.0

    date_str = datetime.fromtimestamp(latest).strftime("%Y%m%d")
    zip_path = zip_path_template.format(date=date_str)

    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for abs_path, arc_path in file_entries:
            zipf.write(abs_path, arc_path)

    print(f"Created: {zip_path}")
    return latest


def build_release_zips(release_dir):
    """Generates downloadable asset zip archives in release_dir."""
    os.makedirs(release_dir, exist_ok=True)
    print(f"Generating release archives in {release_dir}...")

    # Fonts
    create_zip(
        os.path.join(BASE_DIR, "public", "fonts"),
        os.path.join(release_dir, "iw3_player_fonts_{date}.zip"),
        "public/fonts",
    )

    # Lib (Vendor JS)
    create_zip(
        os.path.join(BASE_DIR, "public", "lib"),
        os.path.join(release_dir, "iw3_player_lib_{date}.zip"),
        "public/lib",
    )

    # Environments (HDRI/GLB)
    create_zip(
        os.path.join(BASE_DIR, "public", "environments"),
        os.path.join(release_dir, "iw3_player_environments_{date}.zip"),
        "public/environments",
    )

    # LUT
    create_zip(
        os.path.join(BASE_DIR, "public", "lut"),
        os.path.join(release_dir, "iw3_player_lut_{date}.zip"),
        "public/lut",
    )

    # Example
    example_dist = os.path.join(BASE_DIR, "example", "dist")
    example_src = example_dist if os.path.exists(example_dist) else os.path.join(BASE_DIR, "example")
    create_zip(
        example_src,
        os.path.join(release_dir, "iw3_player_example_{date}.zip"),
        "example",
        exclude_extensions=[".mkv", ".mp4", ".webm"] if not os.path.exists(example_dist) else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Build downloadable asset zip archives for iw3/player.")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(BASE_DIR, "release"),
        help="Output directory for asset zip archives (default: iw3/player/release)",
    )
    args = parser.parse_args()

    build_release_zips(args.output_dir)
    print(f"\nDone. Release archives generated in: {args.output_dir}")


if __name__ == "__main__":
    main()
