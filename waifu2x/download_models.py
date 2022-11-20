import requests
import tarfile
from os import path
from tqdm import tqdm
from io import BytesIO


if __name__ == "__main__":
    MODEL_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_pretrained_models_20221109.tar.gz"
    BLOCK_SIZE = 2 * 1024 * 1024

    output_path = path.dirname(__file__)
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, ncols=80)
    content = BytesIO()
    for data in response.iter_content(BLOCK_SIZE):
        content.write(data)
        progress_bar.update(len(data))
    progress_bar.close()
    content.seek(0)
    with tarfile.open(fileobj=content, mode="r|gz") as f:
        f.extractall(path=output_path)
