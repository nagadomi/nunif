import numpy as np
from PIL import Image
import argparse
import os


def parse_hald(image_path):
    """
    Parses a HaldCLUT image. Supports both square and strip formats.
    Standard Hald: Blue is slowest, Red is fastest -> [B][G][R]
    Returns a numpy array with shape (grid_size, grid_size, grid_size, 3)
    in [B][G][R] order.
    """
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    data = np.asarray(img, dtype=np.float32) / 255.0

    # Calculate grid size (N) from total pixels (N^3)
    grid_size = int(np.round((width * height) ** (1 / 3)))

    # Hald data is naturally indexed as [B][G][R]
    # We reshape to (N, N, N, 3)
    lut = data.reshape((grid_size, grid_size, grid_size, 3))
    return lut


def resize_lut(lut, target_size):
    """
    Resizes the LUT using trilinear interpolation.
    """
    src_size = lut.shape[0]
    if src_size == target_size:
        return lut

    print(f"Resizing LUT from {src_size} to {target_size}...")

    # Create target grid coordinates
    x = np.linspace(0, src_size - 1, target_size)
    # meshgrid with indexing='ij' gives us [target_b, target_g, target_r]
    grid_b, grid_g, grid_r = np.meshgrid(x, x, x, indexing="ij")

    # Get the 8 surrounding integer coordinates for each target point
    b0 = np.floor(grid_b).astype(int).clip(0, src_size - 2)
    g0 = np.floor(grid_g).astype(int).clip(0, src_size - 2)
    r0 = np.floor(grid_r).astype(int).clip(0, src_size - 2)
    b1 = b0 + 1
    g1 = g0 + 1
    r1 = r0 + 1

    # Calculate interpolation weights
    wb = grid_b - b0
    wg = grid_g - g0
    wr = grid_r - r0

    # For vectorized interpolation, we expand weights to (target_size, target_size, target_size, 1)
    wb = wb[..., np.newaxis]
    wg = wg[..., np.newaxis]
    wr = wr[..., np.newaxis]

    # Sample the 8 corners
    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]

    # Trilinear interpolation
    c00 = c000 * (1 - wr) + c001 * wr
    c01 = c010 * (1 - wr) + c011 * wr
    c10 = c100 * (1 - wr) + c101 * wr
    c11 = c110 * (1 - wr) + c111 * wr

    c0 = c00 * (1 - wg) + c01 * wg
    c1 = c10 * (1 - wg) + c11 * wg

    result = c0 * (1 - wb) + c1 * wb
    return result


def save_cube(lut, output_path, title):
    """
    Saves the LUT data as a .cube file.
    Iteration order: Red fastest, then Green, then Blue slowest (RGB order).
    """
    grid_size = lut.shape[0]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Created by convert_cube.py\n")
        f.write(f'TITLE "{title}"\n')
        f.write(f"LUT_3D_SIZE {grid_size}\n")
        f.write("\n")

        # Iteration order: Blue (outer), Green, Red (inner/fastest)
        # This matches the instruction: "Red(R) is fastest, then Green(G), then Blue(B) slowest"
        for b in range(grid_size):
            for g in range(grid_size):
                for r in range(grid_size):
                    # lut is in [b][g][r] order from parse_hald
                    rgb = lut[b, g, r]
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

    print(f"Success: .cube file saved to {output_path}")
    print(f"Grid Size: {grid_size}")


def main():
    parser = argparse.ArgumentParser(description="Convert HaldCLUT image to .cube LUT.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input HaldCLUT image")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-s", "--size", type=int, default=33, help="Target grid size (e.g. 33). Default is 33.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    title = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(args.output, title + ".cube")

    try:
        lut = parse_hald(args.input)
        if args.size:
            lut = resize_lut(lut, args.size)
        save_cube(lut, output_file, title)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
