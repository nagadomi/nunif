@rem Install python 3.10
python -m venv venv_nunif_CUDA
call .\venv_nunif_CUDA\Scripts\activate
pip3 install torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
python -m waifu2x.download_models
python -m waifu2x.web.webgen
pause
