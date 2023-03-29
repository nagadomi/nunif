if __name__ == "__main__":
    from os import path
    from ..download_models import main as download_main

    model_dir = path.join(path.dirname(__file__), "..", "pretrained_models")
    if not path.exists(model_dir):
        download_main()

    from .server import main
    main()
