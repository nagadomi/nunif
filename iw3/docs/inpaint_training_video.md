# Video Inpainting Model Training

## 1. Generating the Training Dataset

Training data is generated from the source video files. For source videos, ordinary 2D videos are used.

It is recommended that videos are in 1080p resolution. During generation, videos are internally resized to either 1080p or 720p. If resolution is less than 1080p, the video will be upscaled and may appear blurry. Using videos larger than 1080p only slows down processing and brings no particular advantage.

### Dataset Generation Command

Run the following command for each video to generate training data. It is recommended to first verify the behavior using a short video.

```
python create_training_data.py video_inpaint --dataset-dir path/to/video.mp4 --prefix tears -o ./data/video_inpaint_v1/train/ --model-type Any_B
```

After running the command, a directory for each data record will be generated under `./data/video_inpaint_v1/train/`.
Each directory corresponds to one data record.

The directory name will be `{prefix}_{model_type}_{FILE_HASH}_{SEQUENCE_NO}`. Be careful to avoid conflicts between video files and the combination of `--prefix` and `--model-type`.

![Data output example](https://github.com/user-attachments/assets/4cad2728-9611-479d-8937-a4b77ca0a813)

Each directory contains an image sequence. Files ending with `_C.png` are RGB frames, and files ending with `_M.png` are inpainting masks.

![Video sequence data example](https://github.com/user-attachments/assets/451ebf7a-73c6-4461-89ff-68e427f89bc6)


The training data directory requires both `train/` and `eval/` dicrectory.
`train/` is used for training, and `eval/` is used for evaluation. In this example, data is output to `./data/video_inpaint_v1/train`.

### Command Options List

All options can be viewed with:

```
python create_training_data.py video_inpaint -h
```

Some main options are as follows:

| Option                | Description
|-----------------------|------------------------------------------------------------
| `video_inpaint`       | Calls the data generation script for video inpainting (constant)
| `--dataset-dir`       | Specify the path to the input video file
| `-o`                  | Specify the output directory
| `--prefix`            | Specify a prefix for data. Output will be in the format `{prefix}_{model_type}_{FILE_HASH}_{SEQUENCE_NO}`. It is strongly recommended to always specify a prefix for easy data management.
| `--model-type`        | Specify the depth estimation model. Use the same model name as in iw3.
| `--resolution`        | Resolution for depth. Same as the iw3 option.
| `--max-fps`           | Maximum FPS for sampling. Default is 30.
| `--seq`               | Number of frames in one data record. Default is 16. During training, 12 frames are randomly cropped from this sequence.
| `--skip-first`        | Number of frames to skip at the beginning. Default is 0.
| `--skip-interval`     | Interval between data records. Default is 16.
| `--batch-size`        | Batch size for the depth estimation model. Default is 2.
| `--size`              | Image size. Default is 512. Random crops are performed from images generated at 1080p or 720p. During training, further random cropping to 256x256 tiles.
| `--num-samples`       | Number of random crops for `--size`. Default is 1.

#### Input and Output Related Options

Always specify: `--dataset-dir`, `-o`, and `--prefix`.

#### Depth Estimation Model Related Options

`--model-type` and `--resolution`. You can create multiple variations by combining different options for the same video.

#### Data Quantity Related Options

Frequently used options: `--skip-first`, `--skip-interval`.

For short videos, use `--skip-interval 0`; for long videos, try `--skip-interval 120` to adjust the number of records generated per video. When you have few videos, increasing `--num-samples` to 2 or 4 can multiply the number of data records.

### Selecting Source Videos

It is preferable to use videos of various genres. Since inpainting is mainly performed on backgrounds, you do not need to focus too much on the subject.

- Outdoors (city, forest, sea, sky)
- Indoors (rooms, factories)

Example:
- City walking
- Survival or hiking
- Factory tours, office tours, gyms
- Beach

### Batch Processing

When working with many video files or different depth estimation models, it is recommended to use shell scripts or BAT files.

Example:
```bash
#!/bin/bash -e

OUTPUT_DIR=./data/video_inpaint_v1/train/

python create_training_data.py video_inpaint --dataset-dir /data/videos/shibuya.webm -o ${OUTPUT_DIR} --prefix shibuy1 --model-type Any_B
python create_training_data.py video_inpaint --dataset-dir /data/videos/shibuya.webm -o ${OUTPUT_DIR} --prefix shibuy2 --model-type Any_L --resolution 518 --skip-first 16
python create_training_data.py video_inpaint --dataset-dir /data/videos/shibuya.webm -o ${OUTPUT_DIR} --prefix shibuy3 --model-type ZoeD_Any_N

python create_training_data.py video_inpaint --dataset-dir /data/videos/kamakura.webm -o ${OUTPUT_DIR} --prefix kamakura1 --model-type Any_B
python create_training_data.py video_inpaint --dataset-dir /data/videos/kamakura.webm -o ${OUTPUT_DIR} --prefix kamakura2 --model-type Distill_Any_S --skip-first 16
python create_training_data.py video_inpaint --dataset-dir /data/videos/kamakura.webm -o ${OUTPUT_DIR} --prefix akamkura3 --model-type ZoeD_Any_N

# ...
```

When errors occur, use the prefix to delete unfinished data and retry or skip.

### Creating the `eval/` Data

Evaluation data is used to check progress and evaluate performance during training.

**Best practice:**  
Create `eval/` data from videos that are not used in `train/`.

**Quick method:**  
Move some records(data directory) from `train/` to the `eval/` directory.

The recommended data size for the quick method is about 200.

## 2. Training

Starting training of the video inpainting model using the created dataset.

The command is the same as for image inpainting, but with the `--video` option added.
```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1
```

- `-i`: Dataset directory
- `--model-dir`: Output directory for trained models (evaluation progress samples will be saved under `eval/`)
- `--save-eval-step`: Save interval for evaluation data (1: save all, 20: save once every 20 records)

In video model training, `--batch-size` is always set to 1 internally. Each data record contains 12-frame sequences, so `--batch-size 1` corresponds to the VRAM usage of `--batch-size 12` in image training.

`--backward-step` is the number of gradient accumulation steps. In video model training, gradient accumulation is used instead of mini-batch.


By default,  `--max-epoch` is 200 and cyclical learning rate scheduleris used. Learning rate resets every 40 epochs. Note that model accuracy may temporarily decrease after resets. The initial training is somewhat coarse.

To resume after stopping midway, specify the `--resume` option.
```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1 --resume
```

For further training or fine-tuning, switch to Schedule-Free AdamW optimizer without the cyclical learning rate.
Use `--resume` to use the previous training results as initial weights, and `--reset-state` to initialize the scheduler.

Repeat this command multiple times as needed. In my experience, results improve up to about the 3rd time (about 600 epochs).
```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1 --optimizer adamw_schedulefree --resume --reset-state
```

You can also start by specifying a checkpoint file for a pre-trained model. In this case, the model's initial weights will be loaded from the `--checkpoint-file` rather than from the `--model-dir`.
```
python train.py inpaint -i ./data/video_inpaint_v1 --model-dir models/video_inpaint_v1/ --video --backward-step 8 --save-eval-step 1 --optimizer adamw_schedulefree --checkpoint-file iw3/pretrained_models/hub/checkpoints/iw3_light_video_inpaint_v1_20250919.pth
```

## 3. Using a Trained Model

See https://github.com/nagadomi/nunif/pull/552

## 4. Code Structure

- `create_training_data.py`: calls `iw3/training/inpaint/create_training_data_video.py`
- `train.py`: uses `iw3/training/inpaint/trainer.py` and `iw3/training/inpaint/dataset_video.py`

Please refer to the code for more details.
For general questions about machine learning, please use AI chat services.
