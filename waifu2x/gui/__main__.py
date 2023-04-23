import platform

if platform.system() in "Windows":
    from gui import main
else:
    from .gui import main

if __name__ == "__main__":
    main()
