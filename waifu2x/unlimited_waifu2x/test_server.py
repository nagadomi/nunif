# python3 -m waifu2x.unlimited_waifu2x.test_server
# View at http://localhost:8812/
# Do not use this server in product environments.
import bottle
import argparse
from os import path


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cors", action="store_true",
                    help=("Add CORS header for testing wasm multi-threading."
                          " google-chrome does not work for security reasons (localhost CORS), use firefox"))
args = parser.parse_args()
ROOT_DIR = path.abspath(path.join(path.dirname(__file__), "public_html"))


@bottle.get("/<url:re:.*>")
def static_file(url):
    if not url:
        url = "index.html"
    response = bottle.static_file(url, root=ROOT_DIR)
    if args.cors:
        response.set_header("Cross-Origin-Resource-Policy", "cross-origin")
        response.set_header("Cross-Origin-Embedder-Policy", "require-corp")
        response.set_header("Cross-Origin-Opener-Policy", "same-origin")
    return response


def main():
    bottle.run(host="127.0.0.1", port=8812, backend="waitress", debug=True)


if __name__ == "__main__":
    main()
