My playground.

For the time being, I will make incompatible changes.

## Dependencies

- Python 3.6 or later
- git-lfs (for ./pretrained_models)
- [PyTorch](https://pytorch.org/get-started/locally/)
- See requirement.txt

```
apt-get install python3 python3-pip libmagickwand-dev git git-lfs 
pip3 install torch torchvision
git clone https://github.com/nagadomi/nunif.git
cd nunif
pip3 install -r requirement.txt
```


## waifu2x

The repository includes waifu2x pytorch implementation and pretrained models. Currently only convert command is supported.

```
./waifu2x -h
```
