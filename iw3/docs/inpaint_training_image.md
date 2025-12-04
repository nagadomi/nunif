# Image Inpainting Model Training

## 1. Generating the Training Dataset

Training data is generated from the source image files. For source images, ordinary 2D videos are used.


### Preparing the Image Dataset

The source image dataset should have the following directory structure:

```
image_dataset/
├── train/
└── eval/
```

`train/` is the directory for training image data.  
`eval/` is the directory for evaluation (validation) image data.


It is recommended that the shorter side of images is at least 1080px. If the image is smaller, it will be upscaled. (This might be changed later.)

### Dataset Generation Command


Run the following command to generate the training data:

```
python create_training_data.py inpaint --dataset-dir /path/to/image_dataset --prefix image1 -o ./data/inpaint_v1/ --model-type Any_B
```

After executing the command, each dataset will be created under `./data/inpaint_v1/train/` and `./data/inpaint_v1/eval/`. The file names will be in the format `{prefix}_{model_type}_{SEQUENCE_NO}_(M|C).png`. Files ending with `_C.png` are RGB frames, and files ending with `_M.png` are inpainting masks. Be careful to avoid conflicts between `--prefix` and `--model-type`.

### Command Options List

All options can be viewed with:

```
python create_training_data.py inpaint -h`
```
Some main options are as follows:

| Option                | Description
|-----------------------|------------------------------------------------------------
| `inpaint`             | Calls the script for image data creation (constant)
| `--dataset-dir`       | Specify the path to the the input image dataset
| `-o`                  | Specify the output folder
| `--prefix`            | Specify a prefix for data (required)
| `--model-type`        | Specify the depth estimation model. Use the same model name as in iw3.
| `--resolution`        | Resolution for depth. Same as the iw3 option.
| `--size`              | Image size. Default is 512. During training, further random cropping to 256x256 tiles.
| `--num-samples`       | Number of random crops for `--size`. Default is 2.

### Batch Processing

When generating data from multiple image datasets or different combinations of depth estimation models, it is recommended to use a shell script or BAT file.

Example:
```bash
#!/bin/bash -e

OUTPUT_DIR=./data/inpaint_v1/

python create_training_data.py inpaint --dataset-dir /data/Flickr2K -o ${OUTPUT_DIR} --prefix flickr1 --model-type Any_B
python create_training_data.py inpaint --dataset-dir /data/Flickr2K -o ${OUTPUT_DIR} --prefix flickr2 --model-type ZoeD_Any_N
# ...
```
When errors occur, use the prefix to delete unfinished data and retry or skip.

## 2. Training

Starting training of the image inpainting model using the created dataset.

```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_v1/
```

- `-i`: Dataset directory
- `--model-dir`: Directory to output the model (training progress samples are saved under `eval/`)
- `--save-eval-step`: Save interval for evaluation data (default: 20. 1: save all, 20: save every 20 batches)

By default,  `--max-epoch` is 200 and cyclical learning rate scheduleris used. Learning rate resets every 40 epochs. Note that model accuracy may temporarily decrease after resets.

To resume after stopping midway, specify the `--resume` option.

```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_v1/ --resume
```

For further training or fine-tuning, switch to Schedule-Free AdamW optimizer without the cyclical learning rate.
Use `--resume` to use the previous training results as initial weights, and `--reset-state` to initialize the scheduler.

Repeat this command multiple times as needed. In my experience, results improve up to about the 3rd time (about 600 epochs).
```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_v1/ --optimizer adamw_schedulefree --resume --reset-state
```

You can also start by specifying a checkpoint file for a pre-trained model. In this case, the model's initial weights will be loaded from the `--checkpoint-file` rather than from the `--model-dir`.
```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_v1/ --optimizer adamw_schedulefree --checkpoint-file iw3/pretrained_models/hub/checkpoints/iw3_light_inpaint_v1_20250919.pth
```

### Training with GAN

Specifying the `--discriminator` option enables training using GAN. By using adversarial loss, checkerboard artifacts can be reduced.

The fine-tuning command using GAN is as follows:
```
python train.py inpaint -i ./data/inpaint_v1 --model-dir models/inpaint_gan_v1/ --optimizer adamw_schedulefree --max-epoch 30 --discriminator l3cffce --save-epoch --disable-hard-example --checkpoint-file models/inpaint_v1/inpaint.light_inpaint_v1.pth
```

Currently, if `--max-epoch` is too high, the effect of adversarial loss may be lost, so stop at an appropriate point. With `--save-epoch`, a checkpoint file for each epoch will be saved.

## 3. Using a Trained Model

See https://github.com/nagadomi/nunif/pull/552

## 4. Code Structure

- `create_training_data.py`: calls `iw3/training/inpaint/create_training_data.py`
- `train.py`: uses `iw3/training/inpaint/trainer.py` and `iw3/training/inpaint/dataset.py`

Please refer to the code for more details.
For general questions about machine learning, please use AI chat services.
