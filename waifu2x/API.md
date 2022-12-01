# API Document
## Server Information
|Language|Framework|Request Format|
|:---:|:---:|:---:|
|Python|Bottle|Json|

## API

### /
|Method|Status|Contents|Return|
|:---:|:---:|:---:|:---:|
|GET|200|None|main html|

### /api
|Method|Status|Contents|Return|
|:---:|:---:|:---:|:---:|
|POST|200|file=img file, data=`{"url": url, "style": style, "scale": scale, "noise": noise, "format": image_format}`|Waifu2x img|

##### Option
|url|style|scale|noise|format|
|:---:|:---:|:---:|:---:|:---:|
|img url|Artwork=art  Photo=photo|None=-1  1.6x=1  2x=2|None=1  Low=0  Medium=1  High=2  Highest=3|PNG=0  WebP=1|

#### Note
Noise is expect JPEG artifact

If you put a file, don't put the url in data .

`{"style": style, "scale": scale, "noise": noise, "format": image_format}`

If you put a url, don't put the file .
