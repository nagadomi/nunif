# python3 -m waifu2x.web.unlimited_waifu2x.test_server
# View at http://localhost:8812/
# Do not use this server in product environments.
import bottle
from os import path


ROOT_DIR = path.abspath(path.join(path.dirname(__file__), "public_html"))


@bottle.get("/<url:re:.*>")
def static_file(url):
    if not url:
        url = "index.html"
    return bottle.static_file(url, root=ROOT_DIR)


def main():
    bottle.run(host="127.0.0.1", port=8812, backend="waitress", debug=True)


if __name__ == "__main__":
    main()
