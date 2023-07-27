@rem Install python 3.10
python -m venv venv
call .\venv\Scripts\activate
pip install -r requirements-torch.txt
pip install -r requirements.txt
pip install -r requirements-gui.txt
python -m waifu2x.download_models
python -m waifu2x.web.webgen
pause
