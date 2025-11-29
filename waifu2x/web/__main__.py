if __name__ == "__main__":
    from os import path
    from ..download_models import main as download_main
    from .webgen.gen import main as webgen_main
    from ..model_dir import MODEL_DIR
    from .public_dir import PUBLIC_DIR

    if not path.exists(MODEL_DIR):
        download_main()

    if not path.exists(PUBLIC_DIR):
        webgen_main()

    from .server import main
    main()
