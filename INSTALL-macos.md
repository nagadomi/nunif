## 1. Install dependencies packages

Install Python 3.10+

```
brew install python3
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
python3 -m venv .venv
```

activate
```
source .venv/bin/activate
```

## 4. Install PyTorch and pip packages

```
pip3 install -r requirements-torch.txt
pip3 install -r requirements.txt
pip3 install -r requirements-gui.txt
```
