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
