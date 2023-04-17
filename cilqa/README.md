`cilqa` provides low-vision image quality scores that are more consistent across different images.

It is useful for filtering low-quality images with a threshold value.

# JPEGQuality

Content-based JPEG Quality predictor.

This can detect low-quality JPEG images even in the following cases.

- JPEG image in PNG format
- Re-encoded JPEG image with the higher quality value than the original quality value 

The current pre-trained model can predict JPEG quality value around an average error of 2/100 on clean validation data.
In real world data it should be much worse. However, I feel that it has achieved a practical level.

## cilqa.filter_low_quality_jpeg

This tool copies only high quality images in the directory specified by `-i` to the directory specified by `-o`.
```
pytyon -m cilqa.filter_low_quality_jpeg -i /path_to_input_dir -o /path_to_output_dir
```

options.
```
options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        input image directory (default: None)
  --output OUTPUT, -o OUTPUT
                        output image directory (default: None)
  --checkpoint CHECKPOINT
                        model parameter file (default: ./cilqa/pretrained_models/jpeg_quality.pth)
  --gpu GPU [GPU ...], -g GPU [GPU ...]
                        GPU device ids. -1 for CPU (default: [0])
  --num-patches NUM_PATCHES
                        number of 128x128 patches used per image (default: 8)
  --quality QUALITY     quality threshold (default: 90)
  --filter-420          drop all 4:2:0(chroma-subsampled) JPEG images (default: False)
  --symlink             create symbolic links, instead of copying the real files (recommended on linux) (default: False)
  --score-prefix        add score prefix to the output filename (default: False)
```
