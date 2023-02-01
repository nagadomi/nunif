## 1. Install dependencies packages

```
apt-get install git-core libmagickwand-dev libsnappy-dev libraqm-dev
```

## 2. Clone

```
git clone https://github.com/nagadomi/nunif.git
cd nunif
```

If you want to use the `dev` branch, execute the following command.
```
git clone https://github.com/nagadomi/nunif.git -b dev
```
or
```
git fetch --all
git checkout -b dev origin/dev
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

(exit)
```
deactivate
```

## 4. Install Pytorch

See [Pytorch](https://pytorch.org/get-started/locally/)

```
pip3 install torch torchvision torchaudio torchtext
```

## 5. Install pip packages

```
pip3 install -r requirements.txt
```

## 6. Run waifu2x.web

Generate `waifu2x/web/public_html`
```
python -m waifu2x.web.webgen.gen
```

Start the web server.
```
python -m waifu2x.web
```
Open http://localhost:8812/

If you don't have an NVIDIA GPU, specify the `--gpu -1` option. (CPU Mode)
```
python -m waifu2x.web --gpu -1
```
