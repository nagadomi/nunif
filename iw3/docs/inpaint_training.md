# Inpainting Model Training

This document describes the training procedure for the inpainting model.
The training procedure is largely divided into two phases.

1. Generating training data (`create_training_data.py`)
2. Model Training (`train.py`)

In **Phase 1**, generating a training dataset from prepared images or videos.  
In **Phase 2**, training the inpainting model using the data created in Phase 1.

## Installing Development Modules

First, install the required development modules.
```
python -m pip install -r requirements-dev.txt
```
For Windows, execute it on `nunif-prompt.bat`.

## Difference Between Image and Video Inpainting Models

Image inpainting and video inpainting models use different data formats and model architectures.
There are also some parts in common, but if you only need to target videos, you do not need to train the image inpainting model.

Refer to the links below for each procedure:

- [Video Inpainting Model Training](./inpaint_training_video.md)
- [Image Inpainting Model Training](./inpaint_training_image.md)

## Training Data Generation Method

For training data generation, inpainting masks are generated from standard 2D images.

- **Input data (X):** Images with masked regions
- **Ground truth data (Y):** Original images

The mask is generated to simulate missing regions (occlusion area, hole) using forward warping, not randomly.

### Steps for Mask Generation (for right view case)

1. Warp the depth map for the **left** view using the depth map of the original image.
2. Warp the empty tensor for the **right** view using the depth map warped for the left view.
3. At the position of the original image, missing regions by forward warping is generated.
4. The generated missing regions are filled by Closing (morphology)

An image warped for the left view returns to its original position when warped again for the right view using the same parameters.  
This behavior is used to generate missing regions for the original image.
Since the original image contains no distortion or missing regions caused by forward warping, it can be used as ground truth.

By generating missing regions via forward warping, the following domain knowledge can be modeled:

- The mask appears along the boundary between the foreground and background.
- The mask is filled with the background.
- In the right view, the right side of the mask corresponds to the background and the left side corresponds to the foreground.

This makes it possible to create training data that simulates a realistic hole inpainting task from ordinary 2D images.

![mask-example](https://github.com/user-attachments/assets/12a02a8e-e73c-4aea-97b1-e054d1fe92d7)
