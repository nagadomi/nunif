# ImageNet

Training code for ImageNet.

The purpose of this code is to pre-training feature extractors, discriminators, perceptual loss, etc.

# Training

Use `train.py`.

Example command
```
python train.py imagenet --arch torchvision.vgg11_bn --data-dir /data/dataset/ImageNet --model-dir ./models/vgg11 --norm center --num-samples 50000 --batch-size 64
```
You should prepare ImageNet dataset yourself.
