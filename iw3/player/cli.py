from .server import server_main as backend_main, create_parser, set_state_args


def main():
    try:
        from .download_assets import main as download_main

        download_main()
    except ImportError:
        pass

    parser = create_parser()
    args = parser.parse_args()
    set_state_args(args)
    backend_main(args)


if __name__ == "__main__":
    main()
