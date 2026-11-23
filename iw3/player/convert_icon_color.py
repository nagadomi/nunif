import glob
import os
import re
import sys


def convert_icons(target_color):
    # Base directory for icons
    icon_dir = os.path.join(os.path.dirname(__file__), "public", "icons")
    if not os.path.exists(icon_dir):
        print(f"Error: Directory not found: {icon_dir}")
        return

    svg_files = glob.glob(os.path.join(icon_dir, "*.svg"))
    if not svg_files:
        print("No SVG files found in public/icons/")
        return

    # Regex patterns
    # Match fill="color" but NOT fill="none"
    fill_regex = re.compile(r'fill="(?!none)[^"]*"')
    # Match stroke="color" but NOT stroke="none"
    stroke_regex = re.compile(r'stroke="(?!none)[^"]*"')
    # Match color="color" (less common but used sometimes)
    color_regex = re.compile(r'color="[^"]*"')

    count = 0
    for file_path in svg_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = stroke_regex.sub(f'stroke="{target_color}"', content)
            new_content = fill_regex.sub(f'fill="{target_color}"', new_content)
            new_content = color_regex.sub(f'color="{target_color}"', new_content)

            # Special case for icons that might only have currentColor in the <svg> tag
            # If the tag has currentColor, replace it
            new_content = new_content.replace("currentColor", target_color)

            # Only write if content changed
            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                count += 1
                print(f"Converted: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    print(f"\nDone. {count} icons converted to {target_color}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m iw3.player.convert_icon_color <color>")
        print('Example: python -m iw3.player.convert_icon_color "#cccccc"')
        print('Example: python -m iw3.player.convert_icon_color "white"')
        sys.exit(1)

    convert_icons(sys.argv[1])
