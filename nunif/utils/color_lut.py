# Utilities to load, apply, and generate 3D LUTs.
import os
from os import path
import shutil
import torch
import torch.nn.functional as F
from nunif.utils.downloader import ArchiveDownloader


def load_lut(file_path):
    """
    Load .cube file and return as a torch tensor (1, 3, SIZE, SIZE, SIZE).
    """
    size = 0
    data = []
    with open(file_path, mode="r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
            elif line[0].isdigit() or line.startswith("-") or line.startswith("."):
                data.append([float(v) for v in line.split()])

    if not data or size == 0:
        raise ValueError(f"Failed to parse LUT from {file_path}")

    # .cube data is in R-fastest order (R, G, B):
    # R changes fastest, then G, then B.
    # This maps to (B, G, R, 3) in numpy/torch terms where B is the outermost dimension.
    table = torch.tensor(data, dtype=torch.float32).reshape((size, size, size, 3))
    # Current table is (B, G, R, C) where C is (RGB)
    # We need (C, B, G, R) for grid_sample (where B, G, R correspond to z, y, x)
    table = table.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return table


def apply_lut(x, lut):
    """
    Apply 3D LUT to x using grid_sample.
    x: (B, 3, H, W) or (3, H, W) in [0, 1] range.
    lut: (1, 3, 33, 33, 33)
    """
    batch = True
    if x.ndim == 3:
        x = x.unsqueeze(0)
        batch = False

    # (1, 3, H, W) -> (1, H, W, 3) where channels are (Red, Green, Blue)
    grid = x.permute(0, 2, 3, 1)

    # Add D dimension for 3D grid_sample (N, D_out, H_out, W_out, 3)
    grid = grid.unsqueeze(1)

    # Map [0, 1] to [-1, 1]
    # In grid_sample, coordinates (x, y, z) map to dimensions (W, H, D)
    # If x=Red, y=Green, z=Blue, then lut must be (C, Blue, Green, Red)
    grid = grid * 2.0 - 1.0

    # Expand lut to the batch
    if x.shape[0] != lut.shape[0]:
        lut = lut.expand(x.shape[0], *lut.shape[1:])

    # grid_sample (N, C, D, H, W) input and (N, D_out, H_out, W_out, 3) grid
    out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=True)
    out = out.squeeze(2)
    if not batch:
        out = out.squeeze(0)

    return out


VERSION = "20260316"
LUT_DIR = path.join(path.dirname(__file__), "color_lut")
VERSION_FILE = path.join(LUT_DIR, VERSION)
LUT_URL = f"https://github.com/nagadomi/nunif/releases/download/0.0.0/color_lut_{VERSION}.zip"
HDR2SDR_LUT = {
    "pq2bt709": path.join(LUT_DIR, "pq2bt709.cube"),
    "pq2bt601": path.join(LUT_DIR, "pq2bt601.cube"),
    "hlg2bt709": path.join(LUT_DIR, "hlg2bt709.cube"),
    "hlg2bt601": path.join(LUT_DIR, "hlg2bt601.cube"),
}


def _generate_hdr_lut(source_trc, target_space, output_file, size=33):
    # pip install colour-science colour-hdri
    import colour
    import colour_hdri
    import numpy as np
    """
    Generate 3D LUT for HDR to SDR conversion using colour-hdri filmic operator.
    """
    LUT = colour.LUT3D(size=size)
    RGB = LUT.table

    #  EOTF: Decode HDR to linear RGB (BT.2020)
    if source_trc == "PQ":
        # ST 2084 returns 0-10000 nits
        RGB_linear = colour.eotf(RGB, "ST 2084")
        # 100 nits = 1.0. With Reinhard2004, this should give more brightness to highlights.
        RGB_linear /= 100.0
    elif source_trc == "HLG":
        # BT.2100 HLG EOTF returns absolute luminance [0, 1000] nits by default
        RGB_linear = colour.eotf(RGB, "ITU-R BT.2100 HLG")
        # 100 nits = 1.0 was good for HLG.
        RGB_linear /= 100.0
    else:
        raise ValueError(f"Unknown source TRC: {source_trc}")

    # Tone Mapping (Linear space)
    if source_trc == "PQ":
        # Reinhard2004 operator often looks more "natural" for video.
        RGB_tonemapped = colour_hdri.tonemapping_operator_Reinhard2004(
            np.maximum(0, RGB_linear)
        )
        # Saturation boost (Optional):
        # Many players apply gamut mapping or saturation boost for HDR->SDR.
        # Simple saturation boost in linear space:
        # L = 0.2627*R + 0.6780*G + 0.0593*B (BT.2020 weights)
        luma = np.dot(RGB_tonemapped, [0.2627, 0.6780, 0.0593])[..., np.newaxis]
        saturation_factor = 1.2
        RGB_tonemapped = luma + (RGB_tonemapped - luma) * saturation_factor
    else:
        # Keep Filmic for HLG or others for now
        RGB_tonemapped = colour_hdri.tonemapping_operator_Reinhard2004(
            np.maximum(0, RGB_linear)
        )

    # Color Space Conversion: BT.2020 -> Target (Linear)
    if "709" in target_space:
        target_cs_name = "ITU-R BT.709"
    elif "601" in target_space:
        target_cs_name = "ITU-R BT.470 - 625"
    else:
        target_cs_name = target_space

    target_cs = colour.RGB_COLOURSPACES[target_cs_name]
    source_cs = colour.RGB_COLOURSPACES["ITU-R BT.2020"]

    RGB_target_linear = colour.RGB_to_RGB(
        RGB_tonemapped,
        source_cs,
        target_cs
    )

    # OETF: Encode to SDR
    if "709" in target_space:
        RGB_sdr = colour.oetf(np.clip(RGB_target_linear, 0, 1), "ITU-R BT.709")
    elif "601" in target_space:
        RGB_sdr = colour.oetf(np.clip(RGB_target_linear, 0, 1), "ITU-R BT.601")
    else:
        RGB_sdr = np.clip(RGB_target_linear, 0, 1)

    LUT.table = RGB_sdr
    colour.write_LUT(LUT, output_file)


def _gen_all():
    output_dir = LUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # PQ
    _generate_hdr_lut("PQ", "709", path.join(output_dir, "pq2bt709.cube"))
    _generate_hdr_lut("PQ", "601", path.join(output_dir, "pq2bt601.cube"))
    # HLG
    _generate_hdr_lut("HLG", "709", path.join(output_dir, "hlg2bt709.cube"))
    _generate_hdr_lut("HLG", "601", path.join(output_dir, "hlg2bt601.cube"))

    print(f"LUTs generated in {output_dir} directory.")


class LUTlDownloader(ArchiveDownloader):
    def handle(self, src):
        src = path.join(src, "color_lut")
        dst = LUT_DIR
        shutil.copytree(src, dst, dirs_exist_ok=True)
        with open(VERSION_FILE, mode="w") as f:
            f.write(VERSION)


def _download_lut():
    if not path.exists(VERSION_FILE):
        downloder = LUTlDownloader(LUT_URL, name="nunif/utils/color_lut", format="zip")
        downloder.run()


def load_hdr2sdr_lut(name):
    assert name in HDR2SDR_LUT
    _download_lut()
    return load_lut(HDR2SDR_LUT[name])


def _test_lut():
    import torchvision.transforms.functional as TF
    import torchvision.io as IO

    x = IO.read_image("cc0/320/dog.png") / 255.0
    lut = load_hdr2sdr_lut("pq2bt709")
    x = apply_lut(x, lut)
    TF.to_pil_image(torch.clamp(x, 0, 1)).show()


def _bench():
    import time

    N = 300
    B = 1
    # S = (1080, 1920)  # HD 1260 FPS
    S = (2160, 3840)  # 4K 325 FPS
    device = "cuda"

    x = torch.rand((B, 3, *S), device=device)
    lut = load_hdr2sdr_lut("pq2bt709").to(device)

    for i in range(4):
        apply_lut(x, lut)

    t = time.perf_counter()
    torch.cuda.synchronize()
    for i in range(N):
        apply_lut(x, lut)

    torch.cuda.synchronize()
    print(1 / ((time.perf_counter() - t) / (N * B)), "FPS")


if __name__ == "__main__":
    # _gen_all()
    # _download_lut()
    # _test_lut()
    # _bench()
    pass
