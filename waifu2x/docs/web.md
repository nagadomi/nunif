# waifu2x Web Application

Generate `waifu2x/web/public_html`.
```
python -m waifu2x.web.webgen
```

You can also generate minimal WebUI by specifying the `--template minimal` option.
minimal UI is intended for use when waifu2x.web is used as an embedded iframe in Google Colab for example.
```
python -m waifu2x.web.webgen --template minimal
```

The following line starts the Web Server.
```
python -m waifu2x.web
```
The web server starts at http://localhost:8812/ .

Show help.
```
python -m waifu2x.web -h
```

With TTA, debug log print
```
python -m waifu2x.web --tta --debug
```
or
```
DEBUG=1 python -m waifu2x.web --tta --debug
```

Specify HTTP port number and GPU ID.
```
python -m waifu2x.web --port 8813 --gpu 1
```

Remove all size limits for the private server.
```
python -m waifu2x.web --no-size-limit
```

## Use reCAPTCHA

Copy `waifu2x/web/config.ini.sample` to `waifu2x/web/config.ini`
```
cp waifu2x/web/config.ini.sample waifu2x/web/config.ini
```

Edit `site_key` and `secret_key` in `waifu2x/web/config.ini`.

Run
```
python -m waifu2x.web --enable-recaptcha --config waifu2x/web/config.ini
```

## How to publish waifu2x.web instance

#### --bind-addr 0.0.0.0 option

`--bind-addr 0.0.0.0` option will allow requests from all IP addresses.
```
python -m waifu2x.web --bind-addr 0.0.0.0 --port 8812
```
However, this method is not suitable for large scale access. It is suitable for small-scale public access, such as Local Area Network.

#### nginx reverse proxy

`waifu2x.udp.jp` is deployed using nginx reverse proxy.
The advantage of this method is that static files can be served directly from `public_html/` file system.

Examples of configuration files can be found in https://github.com/nagadomi/nunif/tree/master/waifu2x/web/appendix .

Also, in most cases, `/recaptcha_state.json` can be a static file. You can reduce the load on your application server if you download `/recaptcha_state.json` and place it in `public_html/`.
