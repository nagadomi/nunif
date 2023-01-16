import bottle
import posixpath
from os import path


@bottle.get("/<url:re:.*>")
def static_file(url):
    basename = url
    if not basename:
        basename = "index.html"
    return bottle.static_file(basename, root=path.abspath(path.dirname(__file__)))


def main():
    bottle.run(host="127.0.0.1", port=8812, backend="waitress", debug=True)


if __name__ == "__main__":
    main()
