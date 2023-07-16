# A tool to convert .png to .webp.
# WARNING: Note that the original .png will be deleted.
from PIL import Image
import argparse
from os import path
import os
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from nunif.utils.image_loader import list_images


class ConvertWebP(Dataset):
    def __init__(self, input_dir):
        super().__init__()
        self.files = list_images(input_dir, extensions=[".png"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        filename = self.files[i]
        try:
            im = Image.open(filename)
            im.load()
        except:
            return -1
        dirname = path.dirname(filename)
        basename = path.splitext(path.basename(filename))[0]
        output_filename = path.join(dirname, basename + ".webp")
        try:
            im.save(output_filename, lossless=True)
        except:
            if path.exists(output_filename):
                os.unlink(output_filename)
            return -1
        os.unlink(filename)

        return 0


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="input directory or file")
    args = parser.parse_args()
    num_workers = cpu_count()

    loader = DataLoader(
        ConvertWebP(args.input),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=8,
        drop_last=False
    )
    for _ in tqdm(loader, ncols=80):
        pass


if __name__ == "__main__":
    main()
