import os
import logging


def _setup():
    logger = logging.getLogger("nunif")
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s: [%(levelname)8s] %(message)s"))

    debug = os.getenv("DEBUG")
    if debug is not None and debug.isdigit():
        debug = int(debug)
    if bool(debug):
        handler.setLevel(logging.DEBUG)
        logger.setLevel(level=logging.DEBUG)
    else:
        handler.setLevel(logging.INFO)
        logger.setLevel(level=logging.INFO)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = _setup()


def set_log_level(level):
    for handler in logger.handlers[:]:
        handler.setLevel(level)
    logger.setLevel(level=level)
