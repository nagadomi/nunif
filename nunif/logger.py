import os
import logging

# TODO: fix duplicate line when using multi threading

logger = logging.getLogger("nunif")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s"))

if os.getenv("DEBUG") is not None:
    handler.setLevel(logging.DEBUG)
    logger.setLevel(level=logging.DEBUG)
logger.addHandler(handler)
