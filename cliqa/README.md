# CLIQA

`cliqa` provides low-vision image quality scores that are more consistent across different images.

It is useful for filtering low-quality images with a threshold value when creating image datasets.

# JPEGQuality

Content-based JPEG Quality predictor.

This can detect low-quality JPEG images even in the following cases.

- (past) JPEG encoded image in PNG format
- Re-encoded JPEG image with the higher quality value than the original quality value 

The current pre-trained model can predict JPEG quality value around an average error of 3/100 on clean validation data.
In real world data it should be much worse. However, I feel that it has achieved a practical level.

## cliqa.filter_low_quality_jpeg

This tool copies only high quality images in the directory specified with `-i` to the directory specified with `-o`.
```
pytyon -m cliqa.filter_low_quality_jpeg -i /path_to_input_dir -o /path_to_output_dir
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
                        model parameter file (default: ./cliqa/pretrained_models/jpeg_quality.pth)
  --gpu GPU [GPU ...], -g GPU [GPU ...]
                        GPU device ids. -1 for CPU (default: [0])
  --num-patches NUM_PATCHES
                        number of 128x128 patches used per image (default: 8)
  --quality QUALITY     quality threshold (default: 90)
  --filter-420          drop all 4:2:0(chroma-subsampled) JPEG images (default: False)
  --symlink             create symbolic links, instead of copying the real files (recommended on linux) (default: False)
  --score-prefix        add score prefix to the output filename (default: False)
```

The threshold value is specified with `--quality` option (default: 90, it is a predicted JPEG quality).

On Linux, it is recommended to always use `--symlink` option.

The predicted quality of each image can be added to the filename with `--score-prefix` option. Then, sorting by filename is in order of quality.
`--quality 0 --score-prefix` allows the score prefix to be added to all files without filtering out low quality images.

# GrainNoiseLevel

Content-based Photo Noise Level predictor. This can detect noisy photos.

The predicted value is `noise_level = 50 - min(PSNR(original_image, noise_added_image), 50)`, so `predicted_psnr = 50 - predicted_noise_level`.
`predicted_psnr >= 40` is predicted to be a less noisy photo.

Note that this model is trained with not very perfected synthetic noise. May not work well in some real world noise cases.

## cliqa.filter_noisy_photo

This tool copies only non-noisy images in the directory specified with `-i` to the directory specified with `-o`.
```
pytyon -m cliqa.filter_noisy_photo -i /path_to_input_dir -o /path_to_output_dir
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
                        model parameter file (default: ./cliqa/pretrained_models/grain_noise_level.pth)
  --gpu GPU [GPU ...], -g GPU [GPU ...]
                        GPU device ids. -1 for CPU (default: [0])
  --num-patches NUM_PATCHES
                        number of 128x128 patches used per image (default: 8)
  --psnr PSNR           quality threshold (default: 40)
  --symlink             create symbolic links, instead of copying the real files (recommended on linux) (default: False)
  --score-prefix        add score prefix to the output filename (default: False)
```

The threshold value is specified with `--psnr` option (default: 40, it is a predicted PSNR).

Other options are the same as `filter_low_quality_jpeg`.

## Note

If you want to use both `filter_noisy_photo` and `filter_low_quality_jpeg`, run `filter_low_quality_jpeg` first, then `filter_noisy_photo`.
The reason for this is that `filter_noisy_photo` does not use JPEG quality less than 80 for training.

Example,
```
pytyon -m cliqa.filter_low_quality_jpeg -i ./data/images -o ./data/hq1 --symlink
pytyon -m cliqa.filter_noisy_photo -i ./data/hq1 -o ./data/hq2 --symlink
```

More practical example,
```bash
# rename original `images` directory to `images_original` temporary.
mv ./data/images ./data/images_original
# filtering low quality jpegs
pytyon -m cliqa.filter_low_quality_jpeg -i ./data/images_original -o ./data/images_filter_jpeg --symlink
# filtering noisy photos
pytyon -m cliqa.filter_noisy_photo -i ./data/images_filter_jpeg -o ./data/images_filter_noisy --symlink
# Make the original `images` directory link to the cleaned directory. (replace)
ln -s `realpath ./data/images_filter_noisy` ./data/images
```
