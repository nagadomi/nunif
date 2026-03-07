## 1. Install dependencies packages

Install Python 3.12.

```
brew install python@3.12
```

Install ImageMagick and libraqm.
```
brew install imagemagick libraqm
```

## 2. Clone

```
git clone https://github.com/nagadomi/nunif.git
cd nunif
```

## 3. Setup virtualenv (optional)

initialize
```
python3.12 -m venv .venv
```

activate
```
source .venv/bin/activate
```

## 4. Install PyTorch and pip packages

```
python3.12 -m pip install -r requirements-torch.txt
python3.12 -m pip install -r requirements.txt
python3.12 -m pip install -r requirements-gui.txt
```
(If you are using a venv, you can just use `python` command)
