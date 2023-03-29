import os


def set_image_ext(filename, format):
    """
    Note that this function removes file extension(.*) first.
    This may not work with filenames without extensions.
    """
    format = format.lower()
    filename = os.path.splitext(filename)[0]
    if format == "png":
        return filename + ".png"
    elif format == "webp":
        return filename + ".webp"
    elif format in {"jpg", "jpeg"}:
        return filename + ".jpg"
    else:
        raise NotImplementedError(f"{format}")


def filename2key(filename, subdir_level=0, sep="."):
    """
    this function converts filename in subdirectory into a unique key
    that can be restored to the original filepath (without extension).
    example: input_dir/puyo/poyo.png  -> output_dir/puyo.poyo.png
             output_dir/puyo.poyo.png -> input_dir/puyo/poyo.png
    `sep="."` may cause problems with `path.splitext()`.
    But note that using some special symbols for `sep` may not work with Windows file systems.
    """
    def basename_without_ext(filename):
        return os.path.splitext(os.path.basename(filename))[0]

    filename = os.path.abspath(filename)
    if subdir_level > 0:
        subdirs = []
        basename = basename_without_ext(filename)
        for _ in range(subdir_level):
            filename = os.path.dirname(filename)
            subdirs.insert(0, os.path.basename(filename))
        return sep.join(subdirs + [basename])
    else:
        return basename_without_ext(filename)


if __name__ == "__main__":
    print(set_image_ext("poko/piyo.jpg", format="png"))
    print(set_image_ext("poko/piyo.jpg.webp", format="png"))
    print(set_image_ext("poko/piyo", format="png"))

    print(filename2key("piko/puyo/piyo.png", subdir_level=0) + ".png")
    print(filename2key("piko/puyo/piyo.png", subdir_level=1) + ".png")
    print(filename2key("piko/puyo/piyo.png", subdir_level=2) + ".png")
