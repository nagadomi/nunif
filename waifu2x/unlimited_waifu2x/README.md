# unlimited:waifu2x

An experimental [onnxruntime-web](https://github.com/microsoft/onnxruntime/tree/main/js/web) based in-browser version of waifu2x.

http://unlimited.waifu2x.net/

It works on web browser without uploading images to the remote server.

Pros
- No image size limitation (But you have web browser memory limit)
- Supports new art models includes 4x models
- Supports TTA

Cons
- It's very slow, like 90's dial-up internet access
- Modern web browser with WebAssembly support is required

Processing performance could be improved when WebGPU is available; experimental implementations with WebGPU may be available for public use around April 2023.

# Setup

1. Place ONNX models in `public_html/models`.

The pretrained models are available at https://github.com/nagadomi/nunif/releases (`waifu2x_onnx_models_*.zip`).

2. Publish `public_html` with a web server.

For testing purposes, web server can be run with the following command.
```
python3 -m waifu2x.unlimited_waifu2x.test_server
```
Open http://127.0.0.1:8812/ in web brwoser.

An example nginx config file is available at [waifu2x/web/unlimited_waifu2x/appendix/unlimited.waifu2x.net](appendix/unlimited.waifu2x.net).

Note that the size of the onnx file is very large.
It is recommended to use CDN to reduce transfer fees.

