# waifu2x Model Training

This document assumes that the working directory is the root directory of nunif (this repository).

# Current Limitations

Currently, only `--method scale` training is supported. `noise` and `noise_scale` are not implemented yet.

I have confirmed that a model trained with this training code can beat the current pretrained upcunet model.

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
| `noise_scale` | `noise{level}_scale2x.pth`| `noise{level}_scale2x.checkpoint.pth`
| `noise`       | `noise{level}.pth`        | `noise{level}.checkpoint.pth`

`--arch` option is network architecture. `upconv_7` and `vgg_7` are older versions.

|`--arch`            | Supported `--method`
|--------------------|------------------------
| `waifu2x.upcunet`  | `scale`, `noise_scale`
| `waifu2x.cunet`    | `noise`
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
When `--resume --reset-state` is specified, `last_epoch`, `best_loss`, `optimizer`, and `grad_scaler` are reset. model parameters(weight) are not reset.


## benchmark

You can run the benchmark with the following commands. 
```
python3 -m waifu2x.benchmark waifu2x --method scale --model-dir models/waifu2x_mymodel -i /test_image_dir --tile-size 104 --baseline --baseline-filter lanczos 
```
Use the `-i` option to specify the directory where the test images are located.

When checking results with the pretrained model.
```
python3 -m waifu2x.benchmark waifu2x --method scale --model-dir waifu2x/pretrained_models/cunet/art -i /test_image_dir --tile-size 104
```

(`waifu2x.benchmark` command is not organized and may change in the future.)
