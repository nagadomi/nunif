if __name__ == "__main__":
    from os import path

    from ..download_models import has_models
    from ..download_models import main as download_main
    from .public_dir import PUBLIC_DIR
    from .webgen.gen import main as webgen_main

    if not has_models():
        download_main()

    if not path.exists(PUBLIC_DIR):
        webgen_main()

    from .server import main

    main()
