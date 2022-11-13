My playground.

For the time being, I will make incompatible changes.

## Dependencies

- Python 3 (Probably works with Python 3.6 or later)
- [PyTorch](https://pytorch.org/get-started/locally/)
- See requirement.txt

We usually support the latest version. If there are bugs or compatibility issues, we will specify the version.

1. Install [Pytorch](https://pytorch.org/get-started/locally/)

2. Install dependencies packages

```
apt-get install git-core libmagickwand-dev libsnappy-dev
```

3. Clone

```
git clone https://github.com/nagadomi/nunif.git
cd nunif
pip3 install -r requirement.txt
```

## waifu2x

The repository contains waifu2x pytorch implementation and pretrained models.
CLI and Web API is supported. Training is currently not implemented.

See [waifu2x/README.md](waifu2x/README.md)
