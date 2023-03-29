# waifu2x Model Training

This document assumes that the working directory is the root directory of nunif (this repository).

# Current Limitations

- `--seed` option cannot be deterministic for training behavior.

I have confirmed that a model trained with this training code can beat the original pretrained upcunet model.

## create training data

Dataset directory structure
```
/dataset/
├── train/
└── eval/
```
`train/` is image data directory for training. `eval/` is image data directory for evaluation (validation).

The images are supposed to be noise-free PNG images. (`*.jpg`,`*.webp` are also supported as image formats)

Show help.
```
python3 create_training_data.py waifu2x -h
```

Create training data from image dataset.
```
python3 create_training_data.py waifu2x --dataset-dir /data/dataset/waifu2x/ --data-dir ./data/waifu2x
```
This command divides the images into fixed-size images specified by `--size` option(default: 640).
From `--dataset-dir` to `--data-dir`.

The amount of simple images specified by `--reject-rate`(defualt: 0.5) will be rejected. It is intended to reject single-color background areas and to prevent slow training with non hard examples.

## train

Show help.
```
python3 train.py waifu2x -h
```

Basic training command.
```
python3 train.py waifu2x --method scale --arch waifu2x.upcunet --data-dir ./data/waifu2x --model-dir ./models/waifu2x_mymodel
```
Specify with `--data-dir` option the directory created by `create_training_data` command. 
The trained models are saved in the directory specified by `--model-dir`.

| `--method`    | best model file           | latest epoch model file(includes training status)
|---------------|---------------------------|--------------------------------------------------
| `scale`       | `scale2x.pth`             | `scale2x.checkpoint.pth`
| `scale4x`     | `scale4x.pth`             | `scale4x.checkpoint.pth`
| `noise_scale` | `noise{level}_scale2x.pth`| `noise{level}_scale2x.checkpoint.pth`
| `noise_scale4x` | `noise{level}_scale4x.pth` `noise{level}_scale4x.checkpoint.pth`
| `noise`       | `noise{level}.pth`        | `noise{level}.checkpoint.pth`

`--arch` option is network architecture. `upconv_7` and `vgg_7` are older versions.

|`--arch`            | Supported `--method`
|--------------------|------------------------
| `waifu2x.upcunet`  | `scale`, `noise_scale`
| `waifu2x.cunet`    | `noise`
| `waifu2x.swin_unet_1x` | `noise`
| `waifu2x.swin_unet_2x` | `scale`
| `waifu2x.swin_unet_4x` | `scale4x`, `noise_scale4x`
| `waifu2x.upconv_7` | `scale`, `noise_scale`
| `waifu2x.vgg_7`    | `noise`


Few iterations for test run.
```
python3 train.py waifu2x --method scale --arch waifu2x.upcunet --data-dir ./data/waifu2x --model-dir ./models/waifu2x_test --num-samples 500
```
During 1 epoch, the number of data specified by `--num-samples` is used. (default: 50000)

Resume from previous training.
```
python3 train.py waifu2x --method scale --arch waifu2x.upcunet --data-dir ./data/waifu2x --model-dir ./models/waifu2x_mymodel --resume
```
When `--resume` is specified, restores all possible states from the checkpoint file and resumes training.
Note that RNG(Random Number Generator) status cannot be restored.

or load only model parameters and start training from the beginning.
```
python3 train.py waifu2x --method scale --arch waifu2x.upcunet --data-dir ./data/waifu2x --model-dir ./models/waifu2x_mymodel --resume --reset-state
```
When `--resume --reset-state` is specified, `last_epoch`, `best_loss`, `optimizer`, and `grad_scaler` are not loaded. Only the model parameter (weight) is loaded.


Resume from pretraind models.
```
python3 train.py waifu2x --method scale --arch waifu2x.upcunet --data-dir ./data/waifu2x --model-dir ./models/waifu2x_mymodel --checkpoint-file ./waifu2x/pretrained_models/art/scale2x.pth
```
When `--checkpoint-file` is specified, the model parameter (weight) is initialized by the specified model file.


### SwinUNet Note

For SwinUNet models, I used `--size 64` option when training pretrained models.

(`--size 112` by default. If you want to train 4x with `--size` larger than `256`, you must generate larger images with `create_training_data` command.)

Also, `waifu2x.swin_unet_2x`/`waifu2x.swin_unet_4x` sometimes causes NaN. If this problem happens, decrease `--learning-rate`.

(I manually decrease `--learning-rate` after NaN Exception happens. Root solution is to use `--disable-amp` option, but the training process is much slower.)

```
python3 train.py waifu2x --method scale4x --arch waifu2x.swin_unet_4x --data-dir ./data/waifu2x --model-dir ./models/swin --warmup-epoch 1 --loss lbp5 --size 64 --batch-size 16 --optimizer adamw  --learning-rate 0.0001 --resume --reset-state
```

See also [appendix](../appendix/).

### Photo model Note

For Photo model training, specify `--style photo` option. This will affect the synthetic noise image generation during training.

Currently, training with GAN has not been very successful.

See the code below for the script used to train the current photo model.

[train_photo_psnr.sh](../appendix/train_photo_psnr.sh), [train_photo_gan.sh](../appendix/train_photo_gan.sh).

## CLI

You can use the trained models with [CLI](./cli.md) by specifying the trained model directory with `--model-dir` option.

## benchmark

PSNR displayed during training depends on mini-batch size and input image size.
It also displays lower/higher score due to some sort of validation process.

A fair score can be calculated from real input image data with benchmark commands.

Show help.
```
python3 -m waifu2x.benchmark -h
```

You can run the benchmark with the following commands.
```
python3 -m waifu2x.benchmark --method scale --model-dir models/waifu2x_mymodel -i /test_image_dir --color y_matlab --filter catrom --baseline --baseline-filter catrom
```
Use the `-i` option to specify the directory where the test images are located.

Note: `catrom` is bicubic interpolation in ImageMagick.
