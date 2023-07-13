if __name__ == "__main__":
    from os import path
    from ..download_models import main as download_main
    from .webgen.gen import main as webgen_main

    model_dir = path.join(path.dirname(__file__), "..", "pretrained_models")
    if not path.exists(model_dir):
        download_main()

    public_html_dir = path.join(path.dirname(__file__), "public_html")
    if not path.exists(public_html_dir):
        webgen_main()

    from .server import main
    main()
