# Install python 3.10
python -m venv venv
call .\venv\Scripts\activate
pip3 install torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
python -m waifu2x.download_models
python -m waifu2x.web.webgen
pause
