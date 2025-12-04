from os import path
from nunif.utils.home_dir import ensure_home_dir


PUBLIC_DIR = path.join(ensure_home_dir("waifu2x"), "web", "public_html")
